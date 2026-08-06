import argparse, csv, re, time, datetime, os
import serial

FIELDS = ["ts","tag","run","model_banner","head_mb","psram_free_kb","tokens",
          "seconds","tok_per_s","ms_per_tok","ms_input","ms_attn","ms_ffn",
          "ms_ple","ms_head"]

def capture_one(ser, timeout=240):
    row = {}
    ser.reset_input_buffer()
    ser.setDTR(False)
    ser.setRTS(True);  time.sleep(0.1)
    ser.setRTS(False)
    start = last_thr = time.time()
    while time.time() - start < timeout:
        line = ser.readline().decode("utf-8", "replace").strip()
        if not line:
            if "tok_per_s" in row and time.time() - last_thr > 5: break
            continue
        if line.startswith("model:"): row["model_banner"] = line
        m = re.search(r"head staged int8:\s*([\d.]+) MB", line)
        if m: row["head_mb"] = m.group(1)
        m = re.search(r"PSRAM free after alloc:\s*(\d+) KB", line)
        if m: row["psram_free_kb"] = m.group(1)
        m = re.search(r"---\s*(\d+) tokens in ([\d.]+) s ---", line)
        if m: row["tokens"], row["seconds"] = m.group(1), m.group(2)
        m = re.search(r"throughput:\s*([\d.]+) tok/s\s+\(([\d.]+) ms/token\)", line)
        if m:
            row["tok_per_s"], row["ms_per_tok"] = m.group(1), m.group(2)
            last_thr = time.time()
        m = re.search(r"profile ms/token: input ([\d.]+) \| attn ([\d.]+) \| ffn ([\d.]+) \| ple ([\d.]+) \| head ([\d.]+)", line)
        if m:
            row["ms_input"], row["ms_attn"], row["ms_ffn"], row["ms_ple"], row["ms_head"] = m.groups()
            break
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tag",  default="ple-28.9M")
    ap.add_argument("--out",  default="results.csv")
    a = ap.parse_args()
    new = not os.path.exists(a.out)
    ser = serial.Serial(a.port, 115200, timeout=1)
    time.sleep(0.3)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for i in range(a.runs):
            row = capture_one(ser)
            row.update(ts=datetime.datetime.now().isoformat(timespec="seconds"),
                       tag=a.tag, run=i+1)
            w.writerow({k: row.get(k, "") for k in FIELDS}); f.flush()
            print(f"[{a.tag}] run {i+1}/{a.runs}: {row.get('tok_per_s','?')} tok/s")
            time.sleep(2)
    ser.close()

if __name__ == "__main__":
    main()
