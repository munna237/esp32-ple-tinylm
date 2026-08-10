# Flash Is Cheap, Core Is Not

**A controlled latency and memory-hierarchy characterisation of flash-resident
language model inference on an ESP32-S3.**


---

## Attribution

**This repository is a measurement study built on someone else's system.**

The PLE inference runtime, the firmware, the training and export pipeline, and
the model architecture are the work of **Viacheslav Sierbov**
([`slvDev/esp32-ai`](https://github.com/slvDev/esp32-ai)), released under the
MIT licence. His copyright notice is retained in [`LICENSE`](LICENSE) and the
derivation is described in [`NOTICE`](NOTICE).

Our contribution is:

- a controlled benchmark campaign over eleven model configurations, three CPU
  clocks, two flash bus modes, two optimisation levels and three generation
  lengths (497 runs, 492 completed);
- an instrumentation correction to the device firmware, described below, which
  changed measured throughput by up to 56% and invalidated two derived claims;
- a two-parameter analytical latency model, fitted and validated on held-out
  measurements;
- the analysis, figures and write-up.

The inference kernels are unmodified. Our firmware changes add build-config
self-reporting, temperature logging, and the correction in
[Instrumentation correction](#instrumentation-correction).

The Per-Layer-Embedding idea originates in Google's Gemma 3n. The training
corpus is TinyStories (Eldan & Li, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)).

---

## What we found

**Model capacity placed in flash is nearly free; capacity placed in the compute
core is expensive.**

At a fixed core budget, growing the flash-resident embedding table eightfold
improves validation loss with no measurable latency cost. Growing the compute
core instead improves loss by a similar amount but costs 57% of throughput.

The two allocations are not merely different points on a trade-off curve —
one strictly dominates the other:

| config | core | table | tok/s @240 | tok/s @80 | val loss |
|---|---|---|---|---|---|
| `c2M`  | 2.00 M | 1.57 M | 7.08 | 2.70 | 2.044 |
| `d256` | 1.50 M | 6.29 M | **8.82** | **3.41** | **2.031** |

`d256` is faster *and* better, at every clock frequency tested. Six of eleven
configurations lie on the quality–throughput Pareto frontier, and every
dominated configuration is one that spent parameters on the core or on model
width rather than on the table.

### Why

A two-term cost model separates compute from memory by exploiting the fact that
the CPU clock can be set independently of the flash and PSRAM interface clocks:

```
t(f) = C · (240/f) + M
```

Fitted on the 240 and 80 MHz points and validated on a **held-out** 160 MHz
point, this predicts per-token latency to **0.48% mean / 0.61% max** error
across all eleven configurations. The compute-bound fraction is
**78.1–82.6%** (mean 80.0%), largely independent of how capacity is allocated.

Because roughly four fifths of the token is compute-bound, and the
flash-resident table contributes only to the remaining fifth, growing the table
barely touches the critical path.

The model is validated over 80–240 MHz. At 40 MHz it over-predicts by
**12.9%** — below 80 MHz the SoC switches from PLL- to crystal-derived
clocking, which alters APB timing and breaks the clock-invariance assumption.

### Memory tier attribution

Halving flash bandwidth (QIO → DIO at 80 MHz) costs under 5% of total latency,
even though **every weight in this runtime is read from flash on every token**:

| stage | reads from | `c3M` | `d256` |
|---|---|---|---|
| PLE | flash | +7.6% | +6.0% |
| FFN | flash | +5.7% | +5.9% |
| attention | flash | +3.6% | +3.4% |
| output head | **PSRAM** | **+0.7%** | **+0.0%** |
| total | | +5.0% | +4.4% |

The output head is the only stage staged into PSRAM at boot, and the only stage
that does not respond. It serves as a null control for the per-stage timers.

### Other results

- **Compiler benefit is a function of kernel mix, not a fixed number.** `-O3`
  over `-Os` ranges from 19.7% to 58.2% on the same board. A two-parameter fit
  (`1.924×` on the int8 output-head kernel, `1.149×` elsewhere) predicts
  held-out configurations to within 0.43%.
- **Latency depends on generation length.** Mean per-token latency is linear in
  N with a per-position increment of 0.155–0.170 ms (KV-cache growth). `d64`
  runs at 9.49 / 8.78 / 7.64 tok/s at N = 100 / 200 / 400. A throughput figure
  without N is under-specified.
- **Two bottleneck regimes.** The 28.9M reference model (25,353-entry vocab) is
  output-head dominated (55.9% of the token); the 4,096-vocab models are FFN
  dominated. The cost model holds across both.
- **Replication.** We measure the 28.9M reference at 9.42 tok/s and 104.1
  ms/step, against 9.5 tok/s and 102.9 ms/step reported upstream — agreement
  within 1.2% on an independent host and toolchain.
- **Reproducibility.** Across 33 configuration–clock cells measured in two
  separate sessions: max deviation **0.137%**, mean 0.047%.

---

## Instrumentation correction

An initial campaign produced invalid results. We report the correction because
it changed headline numbers and propagated into derived claims.

The device firmware overrides the portable output-head routine with a dual-core
int8 kernel, and staged that head using a **compile-time vocabulary constant
(25,353)** instead of the vocabulary field in the model header. For the
4,096-vocab models this scanned 6.19× more rows than required and read past the
end of the tensor when computing logits.

Three symptoms were diagnostic: staged head size identical (3.35 MB) across
models with different vocabularies; head latency invariant to an 8× vocabulary
reduction; free PSRAM depressed by 2.81 MB in every affected configuration.

Correcting the row bound to `min(V, VOCAB_N)`:

| | before | after |
|---|---|---|
| staged head | 3.35 MB | 0.54 MB |
| free PSRAM | 1723 KB | 4478 KB |
| head latency | 76.3 ms | 13.3 ms |
| throughput (`d32`) | 5.69 tok/s | 8.88 tok/s |

**Two derived claims changed as a result:**

1. An apparent *memory wall* at `D=160` — free PSRAM falling to 186 KB and
   generation hanging — was an artefact of the inflated head allocation
   combined with a fixed-size activation buffer. `D=160` runs normally with
   3582 KB free.
2. The measured `-O3` benefit changed in **both directions** depending on
   regime: down from ~46% to ~24% for the 4k-vocab models, up from ~46% to
   58.2% for the reference model.

`results_prefix_ARCHIVE.csv` retains the pre-correction data for the
before/after comparison only. **It should not be used for anything else.**

---

## Repository layout

```
CSE406/
├── README.md
├── LICENSE                 MIT — Sierbov (upstream) + this work
├── NOTICE                  what is upstream, what is ours
├── paper/
│   └── methods_results.tex
├── tools/                  our measurement code
│   ├── campaign.py         build → flash → measure → CSV
│   ├── complete.py         reference model, compiler, N-sweep, bus, 40 MHz
│   ├── patch_fw.py         idempotent firmware instrumentation patch
│   └── analyze.py          figures, LaTeX tables, summary
├── results/
│   ├── results_v2.csv      corrected dataset — 497 runs
│   ├── results_prefix_ARCHIVE.csv    pre-correction, do not use
│   ├── summary.txt
│   └── figures/            *.png, tables.tex
├── firmware/               upstream, with our instrumentation patch
│   ├── esp32_llm/          sketch, vocab.h, display.h, partitions.csv
│   ├── common/llm.h        portable inference core
│   ├── host_verify/        C-vs-PyTorch verification
│   └── bandwidth_bench/    memory-tier microbenchmark
├── training/               upstream training + export pipeline
│   ├── src/                train, export, quantize, model
│   ├── data/prepare.py
│   └── experiments/
└── models/                 exported .bin, golden logits, training metrics
```

---

## Reproducing

**Hardware.** ESP32-S3 with ≥16 MB flash and 8 MB octal PSRAM (we used an
ESP32-S3-WROOM-1-N16R8). **Toolchain.** arduino-esp32 core 3.3.11,
esptool 5.3.1, Python 3.13 with `pyserial`, `esptool`, `matplotlib`.

```bash
python tools/patch_fw.py --dry-run     # inspect the instrumentation patch
python tools/patch_fw.py               # apply (idempotent, backs up first)

python tools/campaign.py probe         # confirm FQBN options for your core
python tools/campaign.py verify        # one build, one model, one run
python tools/campaign.py sweep --session A    # 11 configs × 3 clocks × 5 runs
python tools/campaign.py sweep --session B    # repeat after a cooldown
python tools/campaign.py bus --session A      # QIO vs DIO

python tools/complete.py ref           # 28.9M reference model
python tools/complete.py opt           # -Os vs -O3
python tools/complete.py ngen          # N = 100 / 400
python tools/complete.py cpu40         # outside the model's validated range

python tools/analyze.py                # figures + tables + summary
```

Every run records the firmware's **self-reported** build tag, CPU clock and
flash clock, so a stale build cannot be silently mislabelled. Runs that fail
are written with an explicit status rather than omitted — this is how we caught
both the watchdog failures and the fact that the 120 MHz flash setting was
never honoured by the module.

---

## Limitations

- **One board, one device.** Device-to-device variation is not characterised.
- **One architecture family.** All configurations are PLE transformers from a
  single codebase; we do not compare against dense or recurrent baselines
  on-device.
- **No energy measurement.** We report latency only. Joules per token on
  MCU-class hardware remains open, and is the obvious next step.
- **Quality is relative, not absolute.** Validation loss is comparable within
  our 4,096-vocab family (shared tokenizer, matched training budget) but not
  across tokenizers, and these models are far too small for instruction
  following or question answering.
- **Single seed.** Each configuration was trained once.
- **Watchdog marginality.** Five of 497 runs failed, all at `c3M @ 80 MHz`,
  where an eight-token watchdog feed interval approaches the 5 s FreeRTOS
  timeout once position-dependent attention growth is accounted for. Reported
  values use the successful runs.

---

## Licence

MIT. See [`LICENSE`](LICENSE) — it retains Viacheslav Sierbov's copyright for
the upstream work and adds ours for the measurement code and analysis.
