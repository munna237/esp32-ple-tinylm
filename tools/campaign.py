"""ESP32-S3 PLE TinyLM benchmark campaign: build, flash, measure, log."""
import argparse, csv, datetime, glob, os, re, subprocess, sys, time
import serial

PORT = "COM5"
MODELS = r"C:\bench\models"
OUT = "results_v2.csv"
BASE = "UploadSpeed=921600,CDCOnBoot=default,FlashSize=16M,PartitionScheme=custom,PSRAM=opi"
CLOCKS = [("cpu240", "CPUFreq=240,FlashMode=qio"),
          ("cpu160", "CPUFreq=160,FlashMode=qio"),
          ("cpu80",  "CPUFreq=80,FlashMode=qio")]
BUSES = [("qio80", "CPUFreq=240,FlashMode=qio"),
         ("dio80", "CPUFreq=240,FlashMode=dio")]
ZOO = ["ple-d32-s0", "ple-d64-s0", "ple-d128-s0", "ple-d256-s0",
       "ple-c1M-s0", "ple-c2M-s0", "ple-c3M-s0",
       "ple-L4-s0", "ple-L8-s0", "ple-D96-s0", "ple-D160-s0"]
BUS_MODELS = ["ple-c3M-s0", "ple-d256-s0"]
FIELDS = ["ts", "session", "tag", "run", "status", "fw_tag", "cpu_mhz", "flash_hz",
          "flash_mode", "model_banner", "head_mb", "psram_free_kb", "temp_start",
          "temp_end", "tokens", "seconds", "tok_per_s", "ms_per_tok", "ms_input",
          "ms_attn", "ms_ffn", "ms_ple", "ms_head", "ms_residual"]

def find_cli():
    for p in glob.glob(r"C:\Program Files\Arduino IDE\**\arduino-cli.exe", recursive=True):
        return p
    sys.exit("arduino-cli.exe not found")

def find_sketch():
    for p in glob.glob("**/esp32_llm.ino", recursive=True):
        return os.path.dirname(os.path.abspath(p))
    sys.exit("esp32_llm.ino not found")

CLI, SKETCH = find_cli(), find_sketch()

def build_upload(label, opts):
    fqbn = "esp32:esp32:esp32s3:" + BASE + "," + opts
    bp = os.path.join(os.environ["TEMP"], "esp32-build-" + label)
    print("\n=== BUILD " + label + " ===\n    " + fqbn)
    r = subprocess.run([CLI, "compile", "--fqbn", fqbn, "--build-property",
                        "compiler.optimization_flags=-O3", "--build-path", bp, SKETCH])
    if r.returncode: print("BUILD FAILED"); return False
    r = subprocess.run([CLI, "upload", "-p", PORT, "--fqbn", fqbn,
                        "--input-dir", bp, SKETCH])
    if r.returncode: print("UPLOAD FAILED"); return False
    time.sleep(2); return True

def flash_model(name):
    b = os.path.join(MODELS, name + ".bin")
    if not os.path.exists(b): print("  MISSING " + b); return False
    print("  -- flashing " + name)
    r = subprocess.run([sys.executable, "-m", "esptool", "--chip", "esp32s3", "--port",
                        PORT, "--baud", "921600", "write_flash", "0x110000", b],
                       stdout=subprocess.DEVNULL)
    if r.returncode: print("  MODEL FLASH FAILED"); return False
    time.sleep(2); return True

def capture(ser, timeout=300, echo=False):
    row = {"status": "TIMEOUT"}
    ser.reset_input_buffer()
    try: ser.setDTR(False); ser.setRTS(True); time.sleep(.12); ser.setRTS(False)
    except AttributeError: ser.dtr = False; ser.rts = True; time.sleep(.12); ser.rts = False
    t0 = time.time()
    while time.time() - t0 < timeout:
        try: ln = ser.readline().decode("utf-8", "replace").strip()
        except Exception: continue
        if not ln: continue
        if echo: print("   | " + ln)
        if ln.startswith("fw: "): row["fw_tag"] = ln[4:]
        elif ln.startswith("cpu_mhz: "): row["cpu_mhz"] = ln.split(": ")[1]
        elif ln.startswith("temp_start: "): row["temp_start"] = ln.split(": ")[1]
        elif ln.startswith("temp_end: "): row["temp_end"] = ln.split(": ")[1]
        elif ln.startswith("model:"): row["model_banner"] = ln
        elif ln.startswith("flash_hz: "):
            m = re.search(r"flash_hz:\s*(\d+)\s+flash_mode:\s*(\d+)", ln)
            if m: row["flash_hz"], row["flash_mode"] = m.groups()
        elif "head staged int8" in ln:
            m = re.search(r"([\d.]+)\s*MB", ln)
            if m: row["head_mb"] = m.group(1)
        elif "PSRAM free after alloc" in ln:
            m = re.search(r"(\d+)\s*KB", ln)
            if m: row["psram_free_kb"] = m.group(1)
        else:
            m = re.search(r"---\s*(\d+) tokens in ([\d.]+) s ---", ln)
            if m: row["tokens"], row["seconds"] = m.groups(); continue
            m = re.search(r"throughput:\s*([\d.]+) tok/s\s*\(([\d.]+) ms/token\)", ln)
            if m: row["tok_per_s"], row["ms_per_tok"] = m.groups(); continue
            m = re.search(r"input ([\d.]+) \| attn ([\d.]+) \| ffn ([\d.]+) \| "
                          r"ple ([\d.]+) \| head ([\d.]+)", ln)
            if m:
                k = ["ms_input", "ms_attn", "ms_ffn", "ms_ple", "ms_head"]
                for i, v in enumerate(m.groups()): row[k[i]] = v
                try:
                    row["ms_residual"] = "%.3f" % (float(row["ms_per_tok"]) -
                                                   sum(float(row[x]) for x in k))
                except Exception: pass
                row["status"] = "OK"
                d = time.time() + 3
                while time.time() < d:
                    t = ser.readline().decode("utf-8", "replace").strip()
                    if echo and t: print("   | " + t)
                    if t.startswith("temp_end: "): row["temp_end"] = t.split(": ")[1]; break
                    if "RUN COMPLETE" in t: break
                return row
    if "model_banner" in row: row["status"] = "PARTIAL"
    return row

def bench(tag, session, runs, echo=False):
    new = not os.path.exists(OUT)
    ser = serial.Serial(PORT, 115200, timeout=1); time.sleep(.3)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for i in range(runs):
            r = capture(ser, echo=(echo and i == 0))
            r.update(ts=datetime.datetime.now().isoformat(timespec="seconds"),
                     session=session, tag=tag, run=i + 1)
            w.writerow({k: r.get(k, "") for k in FIELDS}); f.flush()
            print("  [%s/%s] run %d/%d: %s tok/s  head=%s MB  psram=%s KB  fw=%s  cpu=%s  [%s]"
                  % (tag, session, i + 1, runs, r.get("tok_per_s", "--"),
                     r.get("head_mb", "--"), r.get("psram_free_kb", "--"),
                     r.get("fw_tag", "?"), r.get("cpu_mhz", "?"), r["status"]))
            time.sleep(1.5)
    ser.close()

def main():
    global PORT, MODELS, OUT
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["probe", "verify", "sweep", "bus"])
    a.add_argument("--port", default=PORT)
    a.add_argument("--session", default="A")
    a.add_argument("--runs", type=int, default=5)
    a.add_argument("--models", default=MODELS)
    a.add_argument("--out", default=OUT)
    g = a.parse_args()
    PORT, MODELS, OUT = g.port, g.models, g.out
    print("cli    : " + CLI + "\nsketch : " + SKETCH + "\nmodels : " + MODELS + "\n")

    if g.mode == "probe":
        subprocess.run([CLI, "board", "details", "--fqbn", "esp32:esp32:esp32s3"])
        print("\nCheck CPUFreq accepts 240/160/80 and note FlashMode values.")
    elif g.mode == "verify":
        if build_upload(*CLOCKS[0]) and flash_model(ZOO[1]):
            bench(ZOO[1], "verify", 1, echo=True)
            print("\nCHECK:  fw=v3-sram-actq   head=0.54 MB (NOT 3.35)")
            print("        psram=~4478 KB (NOT ~1723)   tok/s=~9.3")
    elif g.mode == "sweep":
        t0 = time.time(); n = 0
        for label, opts in CLOCKS:
            if not build_upload(label, opts): continue
            for m in ZOO:
                n += 1
                print("\n[%d/%d] %s @ %s" % (n, len(CLOCKS) * len(ZOO), m, label))
                if flash_model(m): bench(m + "@" + label, g.session, g.runs)
        print("\n=== DONE in %d min ===" % ((time.time() - t0) / 60))
    elif g.mode == "bus":
        for label, opts in BUSES:
            if not build_upload(label, opts): continue
            for m in BUS_MODELS:
                if flash_model(m): bench(m + "@" + label, g.session, g.runs)
        print("\n=== BUS SWEEP DONE ===")

if __name__ == "__main__":
    main()