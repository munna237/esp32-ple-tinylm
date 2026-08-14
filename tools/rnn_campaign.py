"""Benchmark campaign for the recurrent baselines (LSTM / GRU).

Mirrors campaign.py exactly --- same builds, same -O3, same five runs per cell,
same CSV --- but drives firmware/esp32_rnn instead of firmware/esp32_llm, so
the two architecture families are measured under an identical protocol.

    python rnn_campaign.py verify              # one build, one model, one run
    python rnn_campaign.py sweep --session A   # 4 models x 3 clocks   ~30 min
    python rnn_campaign.py sweep --session B   # repeat after cooldown
    python rnn_campaign.py ngen --session A    # N = 100 / 400          ~20 min
    python rnn_campaign.py all --session A

Close the Arduino IDE first. Appends to results_v2.csv alongside the
transformer sessions.
"""
import argparse, glob, os, subprocess, sys, time
import campaign as C

PORT = "COM5"
MODELS = r"C:\bench\models"
OUT = "results_v2.csv"
BASE = ("UploadSpeed=921600,CDCOnBoot=default,FlashSize=16M,"
        "PartitionScheme=custom,PSRAM=opi")
CLOCKS = [("cpu240", "CPUFreq=240,FlashMode=qio"),
          ("cpu160", "CPUFreq=160,FlashMode=qio"),
          ("cpu80",  "CPUFreq=80,FlashMode=qio")]
ZOO = ["lstm-c1M-s0", "lstm-c1.5M-s0", "gru-c1.5M-s0", "lstm-c3M-s0"]
NGEN_MODELS = ["lstm-c1M-s0", "lstm-c1.5M-s0", "lstm-c3M-s0"]

# locate the RNN sketch (campaign.py's CLI/SKETCH point at the transformer)
_hits = glob.glob("**/esp32_rnn.ino", recursive=True)
if not _hits:
    sys.exit("firmware/esp32_rnn/esp32_rnn.ino not found")
SKETCH = os.path.dirname(os.path.abspath(_hits[0]))


def build(label, opts, defines=""):
    fqbn = "esp32:esp32:esp32s3:" + BASE + "," + opts
    bp = os.path.join(os.environ["TEMP"], "rnn-" + label)
    print("\n=== BUILD %s %s ===" % (label, defines))
    cmd = [C.CLI, "compile", "--fqbn", fqbn,
           "--build-property", "compiler.optimization_flags=-O3"]
    if defines:
        cmd += ["--build-property", "compiler.cpp.extra_flags=" + defines]
    cmd += ["--build-path", bp, SKETCH]
    if subprocess.run(cmd).returncode:
        print("BUILD FAILED"); return False
    if subprocess.run([C.CLI, "upload", "-p", PORT, "--fqbn", fqbn,
                       "--input-dir", bp, SKETCH]).returncode:
        print("UPLOAD FAILED"); return False
    time.sleep(2)
    return True


def do_verify():
    if build("verify", CLOCKS[0][1]) and C.flash_model(ZOO[1]):
        C.bench(ZOO[1] + "@cpu240", "verify", 1, echo=True)
        print("\nCHECK:  fw=rnn-v1-n200-w8   head=0.54 MB (same as transformer)")
        print("        psram free ~7611 KB (vs ~4476 for the transformer)")
        print("        recurrent state 6.0 KB, constant in position")


def do_sweep(session, runs):
    t0 = time.time(); n = 0
    for label, opts in CLOCKS:
        if not build(label, opts):
            continue
        for m in ZOO:
            n += 1
            print("\n[%d/%d] %s @ %s" % (n, len(CLOCKS) * len(ZOO), m, label))
            if C.flash_model(m):
                C.bench(m + "@" + label, session, runs)
    print("\n=== RNN SWEEP DONE in %d min ===" % ((time.time() - t0) / 60))


def do_ngen(session, runs):
    # The decisive test: with no KV cache, mean per-token latency should be
    # independent of N, where the transformer grows ~0.17 ms per position.
    for n in (100, 400):
        if not build("n%d" % n, CLOCKS[0][1], defines="-DN_GEN=%d" % n):
            continue
        for m in NGEN_MODELS:
            if C.flash_model(m):
                C.bench("%s@n%d" % (m, n), session, runs)
    print("\n=== RNN NGEN DONE ===")


def main():
    global PORT, MODELS, OUT
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["verify", "sweep", "ngen", "all"])
    a.add_argument("--port", default=PORT)
    a.add_argument("--session", default="A")
    a.add_argument("--runs", type=int, default=5)
    a.add_argument("--models", default=MODELS)
    a.add_argument("--out", default=OUT)
    g = a.parse_args()
    PORT, MODELS, OUT = g.port, g.models, g.out
    C.PORT, C.MODELS, C.OUT = PORT, MODELS, OUT
    print("cli    : " + C.CLI + "\nsketch : " + SKETCH + "\nmodels : " + MODELS + "\n")

    if g.mode == "verify":
        do_verify()
    elif g.mode == "sweep":
        do_sweep(g.session, g.runs)
    elif g.mode == "ngen":
        do_ngen(g.session, g.runs)
    else:
        do_sweep(g.session, g.runs)
        do_ngen(g.session, g.runs)


if __name__ == "__main__":
    main()
