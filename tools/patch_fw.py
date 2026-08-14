"""Apply the campaign instrumentation patch to esp32_llm.ino.

Safe to run more than once: every edit checks whether it is already present
and skips if so. Writes a timestamped backup before changing anything, and
refuses to save if brace balance changes.

    python patch_fw.py            # inspect and apply
    python patch_fw.py --dry-run  # report only, change nothing
"""
import datetime, glob, os, re, shutil, sys

DRY = "--dry-run" in sys.argv


def find_sketch():
    hits = [p for p in glob.glob("**/esp32_llm.ino", recursive=True)]
    if not hits:
        sys.exit("esp32_llm.ino not found under " + os.getcwd())
    if len(hits) > 1:
        print("Multiple sketches found; using the first:")
        for h in hits:
            print("   ", h)
    return os.path.abspath(hits[0])


PATH = find_sketch()
src = open(PATH, encoding="utf-8", errors="replace").read()
orig = src
braces_before = src.count("{") - src.count("}")
report = []


def edit(name, marker, anchor, replacement, required=True):
    """Insert `replacement` in place of `anchor` unless `marker` already present."""
    global src
    if marker in src:
        report.append(("SKIP", name, "already applied"))
        return True
    if anchor not in src:
        report.append(("FAIL" if required else "WARN", name, "anchor not found"))
        return False
    src = src.replace(anchor, replacement, 1)
    report.append(("DONE", name, ""))
    return True


# --- 1. build tag and knobs -------------------------------------------------
edit("1 build tag", "FW_TAG",
     "#define LLM_PROFILE 1",
     '#define LLM_PROFILE 1\n'
     '#define FW_TAG   "v3-sram-actq"   // bump on ANY firmware change\n'
     '#define MARK_PIN 21               // GPIO energy-window marker\n')

# --- 2. internal-SRAM allocator ---------------------------------------------
edit("2 ps_internal", "ps_internal(size_t",
     "static void *ps(size_t n) {",
     "// Internal SRAM: single-cycle, no SPI. Used for the per-token activation\n"
     "// buffer, which both cores read on every output-head row.\n"
     "static void *ps_internal(size_t n) {\n"
     "  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);\n"
     '  if (!p) { Serial.printf("SRAM alloc failed (%u bytes)\\n", (unsigned)n); while (1) delay(1000); }\n'
     "  return p;\n"
     "}\n\n"
     "static void *ps(size_t n) {")

# --- 3. move head_actq to SRAM (THE important one) --------------------------
m = re.search(r"head_actq\s*=\s*\(int8_t \*\)\s*ps\(", src)
if "ps_internal(head_cols)" in src or "ps_internal(head_actq" in src:
    report.append(("SKIP", "3 head_actq -> SRAM", "already applied"))
elif m:
    src = re.sub(r"(head_actq\s*=\s*\(int8_t \*\)\s*)ps\(", r"\1ps_internal(", src, count=1)
    report.append(("DONE", "3 head_actq -> SRAM", ""))
elif "static int8_t head_actq[" in src:
    report.append(("FAIL", "3 head_actq -> SRAM",
                   "head_actq is still a fixed static array - the vocab-cap "
                   "patch from your earlier session is MISSING. Stop and fix."))
else:
    report.append(("FAIL", "3 head_actq -> SRAM", "allocation site not found"))

# --- 4. self-reporting boot banner ------------------------------------------
edit("4 boot banner", "cpu_mhz:",
     'Serial.println("\\n=== ESP32-S3 PLE TinyLM ===");',
     'Serial.println("\\n=== ESP32-S3 PLE TinyLM ===");\n'
     '  pinMode(MARK_PIN, OUTPUT);\n'
     '  digitalWrite(MARK_PIN, LOW);\n'
     '  Serial.printf("fw: %s\\n", FW_TAG);\n'
     '  Serial.printf("cpu_mhz: %u\\n", (unsigned)getCpuFrequencyMhz());\n'
     '  Serial.printf("flash_hz: %u  flash_mode: %d\\n",\n'
     '                (unsigned)ESP.getFlashChipSpeed(), (int)ESP.getFlashChipMode());\n'
     '  Serial.printf("temp_start: %.1f\\n", temperatureRead());')

# --- 5. end-of-run temperature + completion marker ---------------------------
edit("5 run footer", "RUN COMPLETE",
     "  blink(0);\n}",
     '  Serial.printf("temp_end: %.1f\\n", temperatureRead());\n'
     '  Serial.println("=== RUN COMPLETE ===");\n'
     "  blink(0);\n}")

# --- 6. GPIO window marker ---------------------------------------------------
edit("6a marker high", "digitalWrite(MARK_PIN, HIGH)",
     "  t_start = esp_timer_get_time();",
     "  digitalWrite(MARK_PIN, HIGH);\n  t_start = esp_timer_get_time();")

edit("6b marker low", "digitalWrite(MARK_PIN, LOW);\n  Serial.printf(\"\\n\\n---",
     "  int64_t total_us = esp_timer_get_time() - t_start;",
     "  int64_t total_us = esp_timer_get_time() - t_start;\n"
     "  digitalWrite(MARK_PIN, LOW);", required=False)

# --- report ------------------------------------------------------------------
print("\nsketch: " + PATH + "\n")
w = max(len(n) for _, n, _ in report)
for status, name, note in report:
    print("  [%-4s] %-*s %s" % (status, w, name, note))

fails = [r for r in report if r[0] == "FAIL"]
braces_after = src.count("{") - src.count("}")

print()
if braces_after != braces_before:
    sys.exit("ABORT: brace balance changed (%d -> %d). Nothing written."
             % (braces_before, braces_after))
if fails:
    sys.exit("ABORT: %d required edit(s) failed. Nothing written.\n"
             "Send this output and I will adjust the anchors." % len(fails))
if src == orig:
    print("No changes needed - firmware already patched.")
    sys.exit(0)
if DRY:
    print("DRY RUN - nothing written.")
    sys.exit(0)

bak = PATH + "." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"
shutil.copy2(PATH, bak)
open(PATH, "w", encoding="utf-8").write(src)
print("backup : " + bak)
print("patched: " + PATH)
print("\nNow close Arduino IDE completely, then run:")
print("    python campaign.py verify")
