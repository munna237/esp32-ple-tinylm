# 🧠 PLE TinyLM on ESP32-S3

**A 28.9M-parameter language model running fully on-device on an $8 microcontroller — no cloud, no Wi-Fi, no subscription.**

This repository contains a complete implementation and benchmark suite for running a 4-bit-quantized, Per-Layer-Embedding (PLE) TinyLM on the **ESP32-S3** (dual-core Xtensa LX7 @ 240 MHz, 8 MB PSRAM, 16 MB flash).

## ✨ Highlights

- **28.9M params @ 4-bit ≈ 14.9 MB**, stored in a memory-mapped flash partition.
- **≈ 9.5 tok/s** end-to-end on a 240 MHz microcontroller.
- **Dual-core int8 output head** — the output matvec is split across both LX7 cores.
- **Cycle-accurate memory-hierarchy benchmark** that validates the whole design.
- **Domain-invariant throughput** — throughput is independent of the training domain (stories / Shakespeare / industrial logs).

## 🧠 Memory Hierarchy

| Component | Size | Location | Access pattern |
|---|---|---|---|
| Model weights (4-bit) | 14.9 MB | Flash (mmap) | sequential / XIP |
| PLE table | ~25M params | Flash (mmap) | 6 random rows / token |
| Output head (int8) | ~1.5 MB | PSRAM | full scan / token |
| KV-cache + scratch | — | PSRAM | R/W |

## 📐 Bandwidth Cost Model

Per token, the cost model is:

$$t_{tok} \approx t_{head} + 6 \cdot t_{row}$$

where

$$t_{head} = \frac{1.5\ \text{MB}}{BW_{PSRAM}}, \qquad t_{row} = \text{flash random-read latency per PLE row}$$

and the throughput ceiling is:

$$\text{tok/s} \approx \frac{1000}{t_{head} + 6\,t_{row}/1000}$$

The benchmark (`bandwidth_bench.ino`) measures all three terms with the Xtensa cycle counter.

## 📊 Representative Results (ESP32-S3 @ 240 MHz, -O3)

| Model | Vocab | Throughput |
|---|---|---|
| PLE TinyLM 28.9M (4-bit) | 25,353 | **9.47 tok/s** |
| PLE 1.5M-core (4-bit), fixed | 4,096 | **8.72 tok/s** |
| same, pre-instrumentation-fix | 4,096 | 5.7 tok/s |

Throughput is **domain-invariant**: models trained on children's stories, Shakespeare, and synthetic industrial logs all achieve statistically identical throughput at equal architecture.

## 🗂️ Repository Structure

```
├── bandwidth_bench.ino   # cycle-accurate PSRAM / SRAM / flash-random benchmark
├── esp32_llm.ino         # inference engine (dual-core int8 head, mmap flash)
├── vocab.h               # token-id -> UTF-8 blob (VOCAB_N = 25353)
└── common/llm.h          # portable PLE inference core (host-verified vs PyTorch)
```

## 🔧 Build & Flash

1. Use an ESP32-S3 with **8 MB PSRAM / 16 MB flash** and a custom partition table with a `model` data partition (subtype `0x40`) large enough for the 14.9 MB weights.
2. Flash the weights to the `model` partition.
3. Build with `-O3` and **PSRAM enabled**, then flash `esp32_llm.ino`.
4. Open Serial at 115200 baud.

## 🐞 Instrumentation Note

An early instrumentation bug capped the output head to the full tokenizer size (25,353 rows) even for 4,096-vocab models, and used a fixed-size activation buffer. Capping the head to $\min(V, V_{N})$ and allocating the activation dynamically raised 4k-vocab throughput from **5.7 → 8.7 tok/s**.

## 📜 License

MIT (placeholder).
