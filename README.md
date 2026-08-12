# Where the Bytes Live

**A memory-hierarchy characterisation of on-device language model inference on
an ESP32-S3.** Transformer and recurrent families, 648 measured runs.

---

## Attribution

**The transformer half of this repository is built on someone else's system.**

The PLE inference runtime, the ESP32 firmware, the training and export
pipeline, and the transformer architecture are the work of **Viacheslav
Sierbov** ([`slvDev/esp32-ai`](https://github.com/slvDev/esp32-ai)), released
under the MIT licence. His copyright is retained in [`LICENSE`](LICENSE) and the
derivation is itemised in [`NOTICE`](NOTICE).

**The recurrent half is ours** — `train_rnn.py`, `export_rnn.py`, `rnn.h`,
`verify_rnn.c` and `esp32_rnn.ino` were written for this study so that a second
architecture family could be measured under an identical protocol.

Our contribution:

- a controlled benchmark of **15 on-device configurations across two
  architecture families**, over three CPU clocks, two flash bus modes, two
  optimisation levels and three generation lengths — 648 runs, 643 completed;
- an **instrumentation correction** to the upstream firmware that changed
  measured throughput by up to 56% and invalidated two derived conclusions;
- a **two-parameter analytical latency model**, validated on held-out
  measurements across both families to under 0.5%;
- a **causal demonstration** that position-dependent latency is produced by the
  KV cache, established by removing the cache rather than by correlation;
- a complete LSTM/GRU training, export, verification and inference stack for
  microcontroller targets.

Per-Layer Embeddings originate in Google's Gemma 3n. The corpus is TinyStories
(Eldan & Li, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)).

---

## Findings

### 1. Capacity in flash is nearly free; capacity in the core is not

At fixed core budget, growing the flash-resident table eightfold improves
quality with no measurable latency cost. Growing the core instead costs 57% of
throughput for a similar gain. The two allocations do not merely trade off —
one strictly dominates:

| config | core | table | tok/s @240 | @80 | val loss |
|---|---|---|---|---|---|
| `c2M`  | 2.00 M | 1.57 M | 7.08 | 2.70 | 2.044 |
| `d256` | 1.50 M | 6.29 M | **8.82** | **3.41** | **2.031** |

Faster *and* better, at every clock. Six of eleven transformer configurations
lie on the quality–throughput Pareto frontier; every dominated one spent
parameters on the core or on width rather than on the table.

### 2. A two-parameter cost model, validated across both families

Because the CPU clock is selectable independently of the flash and PSRAM
interface clocks, per-token latency separates cleanly:

```
t(f) = C * (240/f) + M
```

Fitted on the 240 and 80 MHz points, validated on a **held-out** 160 MHz point:

| family | configs | compute-bound | held-out error |
|---|---|---|---|
| transformer | 11 | 78.1–82.6% | 0.48% mean, 0.61% max |
| recurrent | 4 | 84.3–85.2% | 0.33% mean, 0.43% max |

Roughly four fifths of each token is compute-bound, which is *why* flash-resident
capacity is cheap. The recurrent family sits higher — no KV-cache traffic.

The model holds over 80–240 MHz. At 40 MHz it over-predicts by 12.9%: below
80 MHz the SoC switches from PLL- to crystal-derived clocking, which alters APB
timing and breaks the clock-invariance assumption.

### 3. The KV cache causes position-dependent latency — shown by removing it

| model | N=100 | N=400 | growth |
|---|---|---|---|
| `c1M` (transformer) | 76.8 ms | 100.0 ms | **+30.2%** |
| `d64` (transformer) | 105.3 ms | 130.9 ms | **+24.3%** |
| `c3M` (transformer) | 188.7 ms | 214.1 ms | **+13.5%** |
| `lstm-c1M` | 70.3 ms | 70.3 ms | **0.0%** |
| `lstm-c1.5M` | 98.9 ms | 98.9 ms | **0.0%** |
| `lstm-c3M` | 176.9 ms | 176.9 ms | **0.0%** |

Flat to three significant figures, across a 4x range of generation length. This
is causal evidence by intervention, not correlation.

### 4. Flash bandwidth barely matters, and the null control confirms it

Halving flash bus width (QIO to DIO at 80 MHz) costs under 5% of total latency,
even though **every weight is read from flash on every token**:

| stage | reads from | `c3M` | `d256` |
|---|---|---|---|
| PLE | flash | +7.6% | +6.0% |
| FFN | flash | +5.7% | +5.9% |
| attention | flash | +3.6% | +3.4% |
| output head | **PSRAM** | **+0.7%** | **+0.0%** |
| total | | +5.0% | +4.4% |

The head is the only PSRAM-resident stage and the only one that does not
respond — a null control for the per-stage timers.

### 5. Architecture trade-off, fully priced

At matched core budget, recurrent models are faster and worse:

| budget | transformer | recurrent | speed | quality |
|---|---|---|---|---|
| 1.0 M | `c1M` 11.83 tok/s, 2.111 | `lstm-c1M` 14.21, 2.799 | +20.2% | +0.688 |
| 1.5 M | `d64` 8.79, 2.073 | `lstm-c1.5M` 10.11, 2.731 | +15.1% | +0.658 |
| 1.5 M | `d64` 8.79, 2.073 | `gru-c1.5M` 10.40, 2.693 | +18.4% | +0.619 |
| 3.0 M | `c3M` 5.07, 2.001 | `lstm-c3M` 5.65, 2.610 | +11.4% | +0.609 |

Working state differs by three orders of magnitude — 3.15 MB of KV cache
against 3.5–8.7 KB of recurrent state, leaving 7611 KB of PSRAM free versus
4476 KB. So the design question becomes concrete: **0.62 nats costs 3.06 MB of
PSRAM, 15% of throughput, and latency that grows 24% over a 400-token
generation.**

`gru-c1.5M` also strictly dominates `lstm-c1.5M` — faster *and* better, since
three gates buy a wider hidden state for the same parameters.

### 6. Compiler benefit is predicted by kernel mix

`-O3` over `-Os` ranges from **19.7%** to **58.2%** on identical hardware.
Fitting separate factors for the head and everything else, on the two extreme
configurations only:

```
t(-Os) = 1.149 * t(non-head) + 1.924 * t(head)
```

predicts held-out configurations to within 0.43%. The head is an int8 dot
product the compiler vectorises well; the int4 mixed-precision paths are not.

### Reproducibility

| family | cells | max deviation | mean |
|---|---|---|---|
| transformer | 33 | 0.137% | 0.047% |
| recurrent | 12 | 0.101% | 0.034% |

Measured across two sessions hours apart, with every firmware image and model
payload rewritten between them.

---

## Instrumentation correction

An initial campaign produced invalid results. We report it because it changed
headline numbers *and* propagated into conclusions.

The firmware's dual-core int8 output head staged rows using a compile-time
vocabulary constant (25,353) instead of the model header's vocabulary field.
For the 4,096-vocab models this scanned 6.19x more rows than required and read
past the end of the tensor. Three symptoms were diagnostic: staged head size
identical across models with different vocabularies; head latency invariant to
an 8x vocabulary reduction; free PSRAM depressed by 2.81 MB everywhere.

| | before | after |
|---|---|---|
| staged head | 3.35 MB | 0.54 MB |
| free PSRAM | 1723 KB | 4478 KB |
| head latency | 76.3 ms | 13.3 ms |
| throughput (`d32`) | 5.69 tok/s | 8.88 tok/s |

**Two derived claims changed:**

1. An apparent *memory wall* at `D=160` was an artefact of the inflated
   allocation plus a fixed-size activation buffer. `D=160` runs normally with
   3582 KB free.
2. The `-O3` benefit moved in **both directions** — down from ~46% to ~24% for
   4k-vocab models, up to 58.2% for the reference model.

`results_prefix_ARCHIVE.csv` retains the pre-correction data for the
before/after comparison only. **It should not be used for anything else.**

---

## Repository layout

```
CSE406/
├── README.md   LICENSE   NOTICE
├── paper/
│   ├── methods_results.tex
│   └── intro_related.tex
├── tools/
│   ├── campaign.py         transformer: build -> flash -> measure -> CSV
│   ├── rnn_campaign.py     recurrent: same protocol
│   ├── complete.py         reference model, compiler, N-sweep, bus, 40 MHz
│   ├── patch_fw.py         idempotent firmware instrumentation patch
│   └── analyze.py          figures, LaTeX tables, summary
├── results/
│   ├── results_v2.csv      648 runs, both families
│   ├── results_prefix_ARCHIVE.csv    pre-correction, do not use
│   ├── summary.txt
│   └── figures/            six figures + tables.tex
├── firmware/
│   ├── common/llm.h        PLE transformer core (upstream)
│   ├── common/rnn.h        LSTM/GRU core (ours)
│   ├── esp32_llm/          transformer sketch (upstream + our patches)
│   ├── esp32_rnn/          recurrent sketch (ours)
│   ├── host_verify/        C-vs-PyTorch verification, both families
│   └── bandwidth_bench/    memory-tier microbenchmark (upstream)
├── training/
│   ├── src/                transformer pipeline (upstream)
│   ├── src/train_rnn.py    recurrent training (ours)
│   ├── src/export_rnn.py   recurrent export (ours)
│   ├── data/prepare.py
│   └── experiments/
└── models/                 exported .bin, golden logits, training metrics
```

---

## Reproducing

**Hardware.** ESP32-S3 with at least 16 MB flash and 8 MB octal PSRAM (we used
an ESP32-S3-WROOM-1-N16R8). **Toolchain.** arduino-esp32 core 3.3.11, esptool
5.3.1, Python 3.13 with `pyserial`, `esptool`, `matplotlib`.

```bash
# firmware instrumentation (idempotent, backs up first)
python tools/patch_fw.py --dry-run
python tools/patch_fw.py

# host verification before touching hardware
gcc -O3 -I firmware/common -o verify     firmware/host_verify/verify.c     -lm
gcc -O3 -I firmware/common -o verify_rnn firmware/host_verify/verify_rnn.c -lm
./verify_rnn models/lstm-c1.5M-s0.bin models/lstm-c1.5M-s0_golden.txt

# transformer campaign
python tools/campaign.py probe
python tools/campaign.py verify
python tools/campaign.py sweep --session A
python tools/campaign.py sweep --session B
python tools/campaign.py bus   --session A
python tools/complete.py ref            # 28.9M reference model
python tools/complete.py opt            # -Os vs -O3
python tools/complete.py ngen           # N = 100 / 400
python tools/complete.py cpu40          # outside the validated range

# recurrent campaign
python tools/rnn_campaign.py verify
python tools/rnn_campaign.py sweep --session A
python tools/rnn_campaign.py sweep --session B
python tools/rnn_campaign.py ngen  --session A

python tools/analyze.py
```

Every run records the firmware's **self-reported** build tag, CPU clock and
flash clock, so a stale build cannot be silently mislabelled. Failed runs are
written with an explicit status rather than omitted — that is how we caught both
the watchdog failures and the fact that the 120 MHz flash setting was never
honoured by the module.

---

## Limitations

- **One board.** Device-to-device variation is not characterised.
- **No energy measurement.** Latency and memory only. Joules per token on
  MCU-class hardware remains open; a GPIO window marker is already present in
  both sketches for future instrumentation.
- **Dense transformer baselines are quality-only.** `baseline` (val 2.102) and
  `bigcore` (1.938) were trained under the same protocol but require runtime
  support the PLE firmware lacks, so they are not measured on-device.
- **Quality is relative.** Validation loss is comparable within our
  4,096-vocab family (shared tokenizer, matched training budget) but not across
  tokenizers. These models are far too small for instruction following or
  question answering.
- **Single seed** per configuration.
- **Watchdog marginality.** Five of 648 runs failed, all at `c3M @ 80 MHz`,
  where an eight-token watchdog feed interval approaches the 5 s FreeRTOS
  timeout once position-dependent attention growth is included. Reported values
  use the successful runs.

---

## Licence

MIT. See [`LICENSE`](LICENSE), which retains Viacheslav Sierbov's copyright for
the upstream work and adds ours for the recurrent stack, measurement code and
analysis.
