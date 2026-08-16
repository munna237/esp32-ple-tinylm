"""Interactive cost model for ESP32-S3 language model inference.

Alternative to docs/index.html for anyone who prefers to extend this in Python.
Reads results/results_v2.csv and models/*_metrics.json directly, so it stays in
sync with the measurement data rather than a baked-in copy.

    pip install streamlit pandas
    streamlit run streamlit_app.py
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

# Calibrated on three transformer generation-length sweeps; reproduces them to 1%.
KV_K = 2.15e-4          # ms per (layer x d_model x position)
DIO_PENALTY = 1.047     # measured 4.4-5.0% at an 80 MHz flash clock, CPU at 240 MHz
CAP = {"flash": 16 * 1024**2, "psram": 8 * 1024**2}

TF = ["c1M", "d32", "d64", "d128", "d256", "c2M", "c3M", "L4", "L8", "D96", "D160"]
RN = ["lstm-c1M", "lstm-c1.5M", "gru-c1.5M", "lstm-c3M"]

st.set_page_config(page_title="Where the Bytes Live", layout="wide")


@st.cache_data
def load(root="."):
    df = pd.read_csv(os.path.join(root, "results/results_v2.csv"))
    df = df[df.status == "OK"].copy()
    df["model"] = df.tag.str.split("@").str[0]
    df["cond"] = df.tag.str.split("@").str[1].fillna("")

    def tps(m, c):
        s = df[(df.model == m) & (df.cond == c)].tok_per_s
        return float(s.mean()) if len(s) else None

    out = {}
    for f in glob.glob(os.path.join(root, "models/*_metrics.json")):
        n = os.path.basename(f).replace("_metrics.json", "")
        key = n.replace("ple-", "").replace("-s0", "")
        if key not in TF + RN:
            continue
        j = json.load(open(f))
        cfg, par = j["config"], j["params"]
        t240, t160, t80 = tps(n, "cpu240"), tps(n, "cpu160"), tps(n, "cpu80")
        if not (t240 and t160 and t80):
            continue
        m240, m80 = 1000 / t240, 1000 / t80
        C = (m80 - m240) / 2
        M = m240 - C
        sub = df[(df.model == n) & (df.cond == "cpu240")]
        out[key] = dict(
            family="tf" if key in TF else "rnn",
            cell=cfg.get("cell", "transformer"),
            d=cfg["d_model"], L=cfg["n_layers"],
            FH=cfg.get("ffn_hidden") or cfg.get("hidden"),
            core=par["core"], table=par.get("table", 0), stream=par["stream"],
            val=j["best_val"],
            bin=os.path.getsize(os.path.join(root, "models/%s.bin" % n)),
            C=C, M=M, t240=t240,
            err=100 * ((1.5 * C + M) - 1000 / t160) / (1000 / t160),
            head=float(sub.head_mb.mean()) * 1e6,
            ps_used=(8192 - float(sub.psram_free_kb.mean())) * 1024,
            stages={k: float(sub["ms_" + k].mean())
                    for k in ["input", "ple", "attn", "ffn", "head"]},
        )
    return {k: out[k] for k in TF + RN if k in out}


def predict(c, f, n, dio):
    t = c["C"] * (240 / f) + c["M"]
    if c["family"] == "tf":
        t += KV_K * c["L"] * c["d"] * (n - 200) / 2 * (240 / f)
    return t * (DIO_PENALTY if dio else 1.0)


def state_bytes(c):
    if c["family"] == "tf":
        return 2 * c["L"] * 512 * c["d"] * 4          # KV cache, fp32, full context
    return (1 if c["cell"] == "gru" else 2) * c["L"] * c["FH"] * 4


def mib(b):
    return "%.2f MiB" % (b / 1024**2) if b >= 1024**2 else "%.1f KiB" % (b / 1024)


DATA = load()

st.title("Where the bytes live")
st.caption(
    "Predicted, not live. The coefficients below are fitted from 648 measured runs on one "
    "ESP32-S3-WROOM-1-N16R8. Nothing here is connected to a board."
)

left, right = st.columns([1, 2], gap="large")

with left:
    key = st.selectbox("Configuration", list(DATA), index=list(DATA).index("d256"))
    clk = st.slider("CPU clock (MHz)", 40, 240, 240, 20)
    ngen = st.slider("Generation length (tokens)", 50, 500, 200, 25)
    dio = st.radio("Flash bus", ["QIO · 4-bit", "DIO · 2-bit"], horizontal=True).startswith("DIO")

c = DATA[key]
t = predict(c, clk, ngen, dio)
st_bytes, head = state_bytes(c), c["head"]

with right:
    a, b, d = st.columns(3)
    a.metric("Throughput", "%.2f tok/s" % (1000 / t))
    b.metric("Per-token latency", "%.1f ms" % t)
    d.metric("Validation loss", "%.3f" % c["val"])

    st.caption(
        "%s core · %s table · %s embedding  |  C=%.1f ms, M=%.1f ms, held-out error %.2f%%"
        % (f"{c['core']:,}", f"{c['table']:,}", f"{c['stream']:,}",
           c["C"], c["M"], c["err"])
    )

    st.progress(min(1.0, c["bin"] / CAP["flash"]),
                text="Flash · model payload %s of 16.00 MiB" % mib(c["bin"]))
    st.progress(min(1.0, c["ps_used"] / CAP["psram"]),
                text="PSRAM · %s of 8.00 MiB measured in use (head %s, %s %s)"
                     % (mib(c["ps_used"]), mib(head),
                        "KV cache" if c["family"] == "tf" else "recurrent state",
                        mib(st_bytes)))
    st.caption("SRAM · 512 KiB, not instrumented in this campaign.")

if clk < 80:
    st.warning(
        "Below 80 MHz the flash interface stops tracking the CPU. Measured at 40 MHz, "
        "the model over-predicts throughput by 12.9 %. Treat this as an upper bound."
    )
if c["family"] == "tf" and not 100 <= ngen <= 400:
    st.warning("Generation length is outside the measured 100–400 token range; "
               "the position term is extrapolated here.")

LABEL = {
    "tf": {"input": "input", "ple": "PLE", "attn": "attention", "ffn": "FFN", "head": "head"},
    # the recurrent firmware reuses these timer slots — see esp32_rnn.ino
    "rnn": {"input": "input", "attn": "input path W_ih·x", "ffn": "recurrent path W_hh·h + cell",
            "ple": "output proj + RMSNorm", "head": "head"},
}
st.subheader("Measured stage split")
st.caption("240 MHz, N=200" + ("" if c["family"] == "tf" else " · recurrent timer slots remapped"))
st.bar_chart(
    pd.DataFrame({"ms per token": {LABEL[c["family"]][k]: v for k, v in c["stages"].items()}}),
    horizontal=True,
)
