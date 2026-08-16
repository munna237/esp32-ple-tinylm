# Project page and interactive cost model

Two ways to publish this. They show the same thing; pick one.

## Option A — GitHub Pages (recommended)

Static HTML, no server, no build step, no dependencies. Loads instantly and
will still work in five years.

1. Copy `docs/` into the repository root:

   ```
   esp32-ple-tinylm/
     docs/
       index.html
       figures/*.png
   ```

2. Repository → Settings → Pages → Source: **Deploy from a branch**,
   Branch: `main`, Folder: **`/docs`**. Save.

3. It goes live at `https://<user>.github.io/esp32-ple-tinylm/` in about a
   minute. No Actions workflow needed.

4. Edit the two `<a href="https://github.com/">` links in `index.html` to point
   at your repository and the paper.

The measured data is inlined in `index.html` as a 4.6 KB JSON object, so the
page has no fetch and works offline. To refresh it after new runs, regenerate
the object and replace the `const DATA = {...}` line.

## Option B — Streamlit Community Cloud

Use this if you would rather extend the tool in Python — it reads
`results/results_v2.csv` and `models/*_metrics.json` directly, so it never goes
stale relative to the data.

1. Put `streamlit_app.py` in the repository root and add `requirements.txt`:

   ```
   streamlit>=1.40
   pandas>=2.0
   ```

2. share.streamlit.io → New app → point at the repo and `streamlit_app.py`.

Trade-off: free Streamlit apps sleep after a period of inactivity and take
roughly 30 seconds to wake. For a link in a data-availability statement that a
reviewer clicks once, that cold start is a real cost, which is why Option A is
the recommendation.

## What the model does

Per-token latency, from the paper's fitted coefficients:

```
t(f, N) = C · (240/f) + M  +  k · L · d · (N − 200)/2 · (240/f)
```

- `C`, `M` are fitted per configuration on the 240 and 80 MHz measurements.
- `k = 2.15e-4` ms per (layer × d_model × position) is calibrated on the three
  transformer generation-length sweeps and reproduces them to 1 %.
- DIO flash multiplies the result by 1.047 (measured at an 80 MHz flash clock
  with the CPU at 240 MHz).

Validation: the model reproduces every measured point to within 1.07 %, worst
case at `c1M`, N=400. Held-out 160 MHz error is 0.48 % mean.

Memory residency uses measured values where they exist — the flash bar is the
actual `.bin` size on disk, PSRAM occupancy is `8192 − psram_free_kb` as
reported by the firmware, and the output head is the reported `head_mb`. The KV
cache and recurrent state are computed analytically (`2·L·S·d·4` and
`2·L·H·4`, or `L·H·4` for GRU) and agree with the figures quoted in the paper.
SRAM was never instrumented, so the page says so rather than estimating.

## Honesty constraints built into the page

- Labelled "Predicted, not live" in the panel header, with a caveat block
  stating no board is connected.
- Warns below 80 MHz, where the model over-predicts throughput by 12.9 %.
- Warns outside the measured 100–400 token range.
- The recurrent firmware reuses the transformer timer field names for different
  work; the stage bar relabels them per the mapping documented at the top of
  `esp32_rnn.ino`.
