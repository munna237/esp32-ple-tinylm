"""Completion experiments for the ESP32-S3 characterisation.

    python complete.py patch      # make N_GEN / WDT_EVERY build-time settable
    python complete.py ref        # (1) 28.9M reference model, 3 clocks   ~20 min
    python complete.py opt        # (2) -Os vs -O3, all configs @240      ~50 min
    python complete.py ngen       # (3) N=100 / 400 length sweep          ~40 min
    python complete.py qio120     # (4) faster flash bus                  ~10 min
    python complete.py cpu40      # (5) 40 MHz, extends clock lever to 6x ~40 min
    python complete.py all        # everything, in order

Run `patch` once before `ngen` or `cpu40`.
Appends to results_v2.csv alongside sessions A and B.
"""
import glob, os, re, shutil, subprocess, sys, time
import campaign as C

PORT = "COM5"
MODELS = r"C:\bench\models"
OUT = "results_v2.csv"
SESS = "C"
BASE = ("UploadSpeed=921600,CDCOnBoot=default,FlashSize=16M,"
        "PartitionScheme=custom,PSRAM=opi")
ZOO = ["ple-d32-s0", "ple-d64-s0", "ple-d128-s0", "ple-d256-s0",
       "ple-c1M-s0", "ple-c2M-s0", "ple-c3M-s0",
       "ple-L4-s0", "ple-L8-s0", "ple-D96-s0", "ple-D160-s0"]

C.PORT, C.MODELS, C.OUT = PORT, MODELS, OUT


def build(label, opts, opt_flag="-O3", defines=""):
    """Build + upload with explicit optimisation flag and optional -D defines."""
    fqbn = "esp32:esp32:esp32s3:" + BASE + "," + opts
    bp = os.path.join(os.environ["TEMP"], "esp32-x-" + label)
    print("\n=== BUILD %s  [%s %s] ===" % (label, opt_flag, defines))
    cmd = [C.CLI, "compile", "--fqbn", fqbn,
           "--build-property", "compiler.optimization_flags=" + opt_flag]
    if defines:
        cmd += ["--build-property", "compiler.cpp.extra_flags=" + defines]
    cmd += ["--build-path", bp, C.SKETCH]
    if subprocess.run(cmd).returncode:
        print("BUILD FAILED"); return False
    if subprocess.run([C.CLI, "upload", "-p", PORT, "--fqbn", fqbn,
                       "--input-dir", bp, C.SKETCH]).returncode:
        print("UPLOAD FAILED"); return False
    time.sleep(2)
    return True


# ---------------------------------------------------------------- patch
def do_patch():
    src = open(C.SKETCH + "\\esp32_llm.ino", encoding="utf-8", errors="replace").read()
    orig = src
    if "N_GEN" not in src:
        anchor = re.search(r'#define FW_TAG\s+"[^"]*"[^\n]*\n', src)
        if not anchor:
            sys.exit("FW_TAG line not found - run patch_fw.py first")
        src = src.replace(anchor.group(0),
            '#ifndef N_GEN\n#define N_GEN 200\n#endif\n'
            '#ifndef WDT_EVERY\n#define WDT_EVERY 8\n#endif\n'
            '#define FW_STR2(x) #x\n#define FW_STR(x) FW_STR2(x)\n'
            '#define FW_TAG   "v4-n" FW_STR(N_GEN) "-w" FW_STR(WDT_EVERY)\n', 1)
        print("  [DONE] N_GEN / WDT_EVERY defines")
    else:
        print("  [SKIP] defines already present")

    if "N_GENERATE = 200" in src:
        src = src.replace("N_GENERATE = 200", "N_GENERATE = N_GEN", 1)
        print("  [DONE] N_GENERATE uses N_GEN")
    else:
        print("  [SKIP] N_GENERATE already parameterised")

    if "(step & 7)" in src:
        src = src.replace("(step & 7)", "(step & (WDT_EVERY - 1))", 1)
        print("  [DONE] watchdog feed uses WDT_EVERY")
    else:
        print("  [SKIP] watchdog feed already parameterised")

    if src == orig:
        print("\nno changes needed"); return
    if src.count("{") - src.count("}") != orig.count("{") - orig.count("}"):
        sys.exit("ABORT: brace balance changed")
    p = C.SKETCH + "\\esp32_llm.ino"
    shutil.copy2(p, p + ".prepatch2.bak")
    open(p, "w", encoding="utf-8").write(src)
    print("\npatched (backup written). Defaults 200/8 are byte-identical to v3.")


# ------------------------------------------------- (1) 28.9M reference model
def do_ref():
    dst = os.path.join(MODELS, "ple-ref28M-s0.bin")
    if not os.path.exists(dst):
        srcs = glob.glob("**/firmware/model/model.bin", recursive=True) or \
               glob.glob("**/model/model.bin", recursive=True)
        if not srcs:
            print("!! 28.9M model.bin not found"); return
        print("copying %s -> %s (14.9 MB)" % (srcs[0], dst))
        shutil.copy2(srcs[0], dst)
    # flash the big model ONCE; firmware uploads do not touch 0x110000
    if not C.flash_model("ple-ref28M-s0"):
        return
    for label, opts in [("cpu240", "CPUFreq=240,FlashMode=qio"),
                        ("cpu160", "CPUFreq=160,FlashMode=qio"),
                        ("cpu80",  "CPUFreq=80,FlashMode=qio")]:
        if build("ref-" + label, opts):
            C.bench("ple-ref28M-s0@" + label, SESS, 5, echo=(label == "cpu240"))


# ------------------------------------------------- (2) compiler optimisation
def do_opt():
    if not build("Os240", "CPUFreq=240,FlashMode=qio", opt_flag="-Os"):
        return
    for m in ZOO:
        if C.flash_model(m):
            C.bench(m + "@Os240", SESS, 5)


# ------------------------------------------------- (3) generation length
def do_ngen():
    for n in (100, 400):
        if not build("n%d" % n, "CPUFreq=240,FlashMode=qio",
                     defines="-DN_GEN=%d" % n):
            continue
        for m in ["ple-d64-s0", "ple-c1M-s0", "ple-c3M-s0"]:
            if C.flash_model(m):
                C.bench("%s@n%d" % (m, n), SESS, 5)


# ------------------------------------------------- (4) faster flash bus
def do_qio120():
    if not build("qio120", "CPUFreq=240,FlashMode=qio120"):
        print("!! qio120 build/upload failed - not all flash parts support it")
        return
    for m in ["ple-c3M-s0", "ple-d256-s0"]:
        if C.flash_model(m):
            C.bench(m + "@qio120", SESS, 5)


# ------------------------------------------------- (5) 40 MHz
def do_cpu40():
    # 40 MHz pushes several configs past the 5 s task-WDT at the default
    # 8-token feed, so feed every 2 tokens here. delay(0) is ~free; the
    # tag records the variant.
    if not build("cpu40", "CPUFreq=40,FlashMode=qio", defines="-DWDT_EVERY=2"):
        return
    for m in ["ple-c1M-s0", "ple-d32-s0", "ple-d64-s0", "ple-L4-s0"]:
        if C.flash_model(m):
            C.bench(m + "@cpu40", SESS, 5)


JOBS = [("patch", do_patch), ("ref", do_ref), ("opt", do_opt),
        ("ngen", do_ngen), ("qio120", do_qio120), ("cpu40", do_cpu40)]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    want = sys.argv[1]
    names = [n for n, _ in JOBS]
    if want == "all":
        t0 = time.time()
        for n, f in JOBS:
            print("\n" + "=" * 60 + "\n  %s\n" % n.upper() + "=" * 60)
            f()
        print("\n=== ALL DONE in %d min ===" % ((time.time() - t0) / 60))
    elif want in names:
        dict(JOBS)[want]()
    else:
        sys.exit("unknown job %r; choose from %s or 'all'" % (want, names))
