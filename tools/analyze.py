"""Analysis for the ESP32-S3 characterisation --- transformer and recurrent.

Reads results_v2.csv (+ *_metrics.json from the model directory) and emits:
    fig1_pareto.png        quality vs throughput, both families
    fig2_roofline.png      compute / clock-invariant split
    fig3_clockscale.png    ms/token vs inverse clock
    fig4_bus.png           per-stage QIO vs DIO, PSRAM head as null control
    fig5_reproducibility.png   session A vs B
    fig6_ngen.png          latency vs generation length --- the KV-cache test
    tables.tex             LaTeX tables
    summary.txt            headline numbers

    python analyze.py
"""
import argparse, csv, glob, json, os, re, statistics as st, sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed.  Run:  pip install matplotlib")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results_v2.csv")
ap.add_argument("--models", default=r"C:\bench\models")
ap.add_argument("--out", default="figures")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
O = lambda f: os.path.join(A.out, f)

TF = ["c1M", "d32", "d64", "d128", "d256", "c2M", "c3M", "L4", "L8", "D96", "D160"]
RN = ["lstm-c1M", "lstm-c1.5M", "gru-c1.5M", "lstm-c3M"]
CLOCKS = [240, 160, 80]
TAG = re.compile(r"^(ple|lstm|gru)-(.+?)-s0(?:@(.+))?$")


def parse(tag):
    """-> (family, config, variant)"""
    m = TAG.match(tag)
    if not m:
        return None, None, None
    fam, cfg, var = m.group(1), m.group(2), m.group(3)
    return ("tf", cfg, var) if fam == "ple" else ("rnn", fam + "-" + cfg, var)


rows = [r for r in csv.DictReader(open(A.csv)) if r["status"] == "OK"]
allrows = list(csv.DictReader(open(A.csv)))
print("loaded %d OK rows of %d from %s" % (len(rows), len(allrows), A.csv))

cells, bus = {}, {}
for r in rows:
    fam, cfg, var = parse(r["tag"])
    if not cfg or not var:
        continue
    if var.startswith("cpu"):
        cells.setdefault((fam, cfg, int(var[3:]), r["session"]), []).append(r)
    elif var.startswith("n") and var[1:].isdigit():
        cells.setdefault((fam, cfg, "n" + var[1:], r["session"]), []).append(r)
    elif var == "Os240":
        cells.setdefault((fam, cfg, "Os", r["session"]), []).append(r)
    elif var.startswith("qio") or var.startswith("dio"):
        bus.setdefault((cfg, var.split("-")[0]), []).append(r)


def agg(cfg, key, field="tok_per_s", sess=None):
    v = []
    for (f, c, k, s), rs in cells.items():
        if c == cfg and k == key and (sess is None or s == sess):
            v += [float(x[field]) for x in rs if x[field]]
    return st.mean(v) if v else None


# ---------------------------------------------------------------- metadata
meta = {}
for p in glob.glob(os.path.join(A.models, "*_metrics.json")):
    d = json.load(open(p))
    fam, cfg, _ = parse(os.path.basename(p).replace("_metrics.json", ""))
    if not cfg:
        continue
    c, pa = d.get("config", {}), d.get("params", {})
    meta[cfg] = dict(
        family=fam, core=pa.get("core"), table=pa.get("table", 0),
        total=pa.get("total"), val=d.get("final_val"), ppl=d.get("final_ppl"),
        D=c.get("d_model"), L=c.get("n_layers"),
        F=c.get("ffn_hidden"), P=c.get("ple_dim"), H=c.get("hidden"),
        cell=c.get("cell"), V=c.get("vocab_size"))
if not meta:
    print("!! no *_metrics.json in %s - quality axis will be blank" % A.models)

for (f, cfg, k, s), rs in cells.items():
    if k == 240 and cfg in meta:
        meta[cfg]["head_mb"] = float(rs[0]["head_mb"] or 0)
        meta[cfg]["psram_kb"] = int(rs[0]["psram_free_kb"] or 0)

ORDER = [c for c in TF if c in meta or agg(c, 240)] + \
        [c for c in RN if c in meta or agg(c, 240)]

# ---------------------------------------------------------------- roofline
fit = {}
for cfg in ORDER:
    a, b, c = agg(cfg, 240), agg(cfg, 160), agg(cfg, 80)
    if not (a and b and c):
        continue
    t240, t160, t80 = 1000 / a, 1000 / b, 1000 / c
    C = (t80 - t240) / 2.0
    M = t240 - C
    fit[cfg] = dict(C=C, M=M, t240=t240, t160=t160, t80=t80,
                    cfrac=100 * C / t240, err=100 * (1.5 * C + M - t160) / t160,
                    fam=meta.get(cfg, {}).get("family", "tf"))

FC = {"tf": "#4c72b0", "rnn": "#c44e52"}
FL = {"tf": "transformer (PLE)", "rnn": "recurrent (LSTM/GRU)"}

# ---------------------------------------------------------------- fig 1
have = [c for c in ORDER if c in meta and meta[c].get("val") and agg(c, 240)]
if have:
    par = [c for c in have if not any(
        agg(o, 240) >= agg(c, 240) and meta[o]["val"] <= meta[c]["val"] and
        (agg(o, 240) > agg(c, 240) or meta[o]["val"] < meta[c]["val"]) for o in have)]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for fam in ("tf", "rnn"):
        xs = [agg(c, 240) for c in have if meta[c]["family"] == fam]
        ys = [meta[c]["val"] for c in have if meta[c]["family"] == fam]
        if xs:
            ax.scatter(xs, ys, s=95, zorder=3, c=FC[fam], label=FL[fam],
                       marker="o" if fam == "tf" else "s",
                       edgecolors="k", linewidths=.4)
    for c in have:
        ax.annotate(c, (agg(c, 240), meta[c]["val"]), fontsize=8,
                    textcoords="offset points", xytext=(7, 4))
    pf = sorted((agg(c, 240), meta[c]["val"]) for c in par)
    ax.plot([p[0] for p in pf], [p[1] for p in pf], "--", c="0.4", lw=1, zorder=2)
    ax.set_xlabel("throughput at 240 MHz (tok/s)")
    ax.set_ylabel("validation loss (lower is better)")
    ax.invert_yaxis(); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Quality vs throughput at matched core budgets")
    fig.tight_layout(); fig.savefig(O("fig1_pareto.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 2
if fit:
    ks = [c for c in ORDER if c in fit]
    Cv = [fit[c]["C"] for c in ks]; Mv = [fit[c]["M"] for c in ks]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(ks, Cv, color="#4c72b0", label="compute (scales with clock)")
    ax.bar(ks, Mv, bottom=Cv, color="#dd8452", label="clock-invariant (memory)")
    for i, c in enumerate(ks):
        ax.text(i, Cv[i] + Mv[i] + 4, "%.0f%%" % fit[c]["cfrac"], ha="center", fontsize=8)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in RN:
            lbl.set_color(FC["rnn"])
    ax.set_ylabel("ms / token at 240 MHz")
    ax.set_title("Roofline decomposition (labels = compute-bound fraction; red = recurrent)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(O("fig2_roofline.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 3
if fit:
    fig, ax = plt.subplots(figsize=(7, 5))
    inv = [240.0 / f for f in CLOCKS]
    for c in [k for k in ORDER if k in fit]:
        ax.plot(inv, [1000 / agg(c, f) for f in CLOCKS], "o-", ms=4, lw=1.1,
                c=FC[fit[c]["fam"]], alpha=.85,
                ls="-" if fit[c]["fam"] == "tf" else "--")
    ax.set_xlabel("240 / f  (inverse relative clock)")
    ax.set_ylabel("ms / token")
    ax.set_title("Latency linear in inverse clock (solid = transformer, dashed = recurrent)")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(O("fig3_clockscale.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 4
if bus:
    stages = ["ms_ple", "ms_ffn", "ms_attn", "ms_head"]
    names = ["PLE\n(flash)", "FFN\n(flash)", "attn\n(flash)", "head\n(PSRAM)"]
    cfgs = sorted({c for c, m in bus if (c, "dio80") in bus and m == "qio80"})
    if cfgs:
        fig, axes = plt.subplots(1, len(cfgs), figsize=(4.4 * len(cfgs), 4.2), squeeze=False)
        for ax, cfg in zip(axes[0], cfgs):
            q = [st.mean(float(r[s]) for r in bus[(cfg, "qio80")]) for s in stages]
            d = [st.mean(float(r[s]) for r in bus[(cfg, "dio80")]) for s in stages]
            pct = [100 * (b - a) / a if a else 0 for a, b in zip(q, d)]
            ax.bar(names, pct, color=["#dd8452"] * 3 + ["#4c72b0"])
            for i, p in enumerate(pct):
                ax.text(i, p + .12, "%+.1f%%" % p, ha="center", fontsize=9)
            ax.axhline(0, c="k", lw=.8)
            ax.set_title(cfg); ax.set_ylabel("latency change, QIO->DIO (%)")
            ax.grid(alpha=.3, axis="y")
        fig.suptitle("Halving flash bandwidth: flash stages respond, PSRAM head does not")
        fig.tight_layout(); fig.savefig(O("fig4_bus.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 5
dev = []
for cfg in ORDER:
    for f in CLOCKS:
        a, b = agg(cfg, f, sess="A"), agg(cfg, f, sess="B")
        if a and b:
            dev.append((cfg + "@" + str(f), 100 * (b - a) / a,
                        meta.get(cfg, {}).get("family", "tf")))
if dev:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(dev)), [d[1] for d in dev], color=[FC[d[2]] for d in dev])
    ax.axhline(0, c="k", lw=.8)
    for y in (.2, -.2):
        ax.axhline(y, c="0.5", ls="--", lw=.8)
    ax.set_xticks(range(len(dev)))
    ax.set_xticklabels([d[0] for d in dev], rotation=90, fontsize=6.5)
    ax.set_ylabel("session B - session A (%)")
    ax.set_title("Between-session reproducibility (dashed +/-0.2%%, n=%d)" % len(dev))
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(O("fig5_reproducibility.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 6
NS = [100, 200, 400]
ng = {}
for cfg in ORDER:
    p100, p200, p400 = agg(cfg, "n100"), agg(cfg, 240), agg(cfg, "n400")
    if p100 and p200 and p400:
        ng[cfg] = [1000 / p100, 1000 / p200, 1000 / p400]
if ng:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg, pts in ng.items():
        fam = meta.get(cfg, {}).get("family", "tf")
        ax.plot(NS, [100 * (p - pts[0]) / pts[0] for p in pts], "o-",
                c=FC[fam], lw=1.4, ms=5,
                ls="-" if fam == "tf" else "--", label=cfg)
    ax.axhline(0, c="k", lw=.8)
    ax.set_xlabel("generation length N (tokens)")
    ax.set_ylabel("mean latency increase vs N=100 (%)")
    ax.set_title("Position-dependent latency: transformers grow, recurrent models do not")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(O("fig6_ngen.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- tables
L = [r"% Table 1: configurations", r"\begin{tabular}{llrrrrrrr}\hline",
     r"Config & family & $D$ & $L$ & $F$/$H$ & $P$ & core & table & val loss\\\hline"]
for c in ORDER:
    m = meta.get(c)
    if not m:
        continue
    L.append(r"%s & %s & %s & %s & %s & %s & %.2fM & %.2fM & %.3f\\" % (
        c.replace("_", r"\_"),
        "transformer" if m["family"] == "tf" else (m.get("cell") or "rnn"),
        m.get("D", "-"), m.get("L", "-"),
        m.get("F") or m.get("H") or "-", m.get("P") or "-",
        (m.get("core") or 0) / 1e6, (m.get("table") or 0) / 1e6,
        m.get("val") or float("nan")))
L += [r"\hline\end{tabular}", "", r"% Table 2: throughput and roofline",
      r"\begin{tabular}{lrrrrrrr}\hline",
      r"Config & 240 & 160 & 80 & $C$ (ms) & $M$ (ms) & comp. & err.\\\hline"]
for c in ORDER:
    if c not in fit:
        continue
    f = fit[c]
    L.append(r"%s & %.2f & %.2f & %.2f & %.1f & %.1f & %.0f\%% & %.2f\%%\\" % (
        c.replace("_", r"\_"), agg(c, 240), agg(c, 160), agg(c, 80),
        f["C"], f["M"], f["cfrac"], f["err"]))
L.append(r"\hline\end{tabular}")
open(O("tables.tex"), "w").write("\n".join(L))

# ---------------------------------------------------------------- summary
S = ["DATA", "  OK rows                : %d" % len(rows),
     "  total rows             : %d  (%d non-OK)" % (len(allrows), len(allrows) - len(rows))]
for fam in ("tf", "rnn"):
    ks = [c for c in fit if fit[c]["fam"] == fam]
    if not ks:
        continue
    S += ["", "ROOFLINE - %s  (fit 240+80, held out 160)" % FL[fam],
          "  configs                : %d" % len(ks),
          "  compute fraction       : %.1f%% - %.1f%%  (mean %.1f%%)" % (
              min(fit[c]["cfrac"] for c in ks), max(fit[c]["cfrac"] for c in ks),
              st.mean(fit[c]["cfrac"] for c in ks)),
          "  held-out error         : max %.2f%%, mean abs %.2f%%" % (
              max(abs(fit[c]["err"]) for c in ks),
              st.mean(abs(fit[c]["err"]) for c in ks))]
for fam in ("tf", "rnn"):
    d = [x for x in dev if x[2] == fam]
    if d:
        S += ["", "REPRODUCIBILITY - %s" % FL[fam],
              "  cells                  : %d" % len(d),
              "  max |deviation|        : %.3f%%" % max(abs(x[1]) for x in d),
              "  mean |deviation|       : %.3f%%" % st.mean(abs(x[1]) for x in d)]
if ng:
    S += ["", "GENERATION-LENGTH DEPENDENCE (N=100 -> 400)"]
    for cfg, pts in ng.items():
        fam = meta.get(cfg, {}).get("family", "tf")
        S.append("  %-14s %-12s %6.1f -> %6.1f ms  (%+.1f%%)" % (
            cfg, FL[fam].split()[0], pts[0], pts[2], 100 * (pts[2] - pts[0]) / pts[0]))
if bus:
    S += ["", "BUS SWEEP (QIO80 -> DIO80)"]
    for cfg, mode in sorted(bus):
        if mode != "qio80" or (cfg, "dio80") not in bus:
            continue
        q = st.mean(float(r["ms_per_tok"]) for r in bus[(cfg, "qio80")])
        d2 = st.mean(float(r["ms_per_tok"]) for r in bus[(cfg, "dio80")])
        h1 = st.mean(float(r["ms_head"]) for r in bus[(cfg, "qio80")])
        h2 = st.mean(float(r["ms_head"]) for r in bus[(cfg, "dio80")])
        S.append("  %-8s total %+.1f%%   head (PSRAM null control) %+.1f%%" % (
            cfg, 100 * (d2 - q) / q, 100 * (h2 - h1) / h1 if h1 else 0))
S += ["", "MATCHED-BUDGET COMPARISON (240 MHz)"]
for t, r in [("c1M", "lstm-c1M"), ("d64", "lstm-c1.5M"),
             ("d64", "gru-c1.5M"), ("c3M", "lstm-c3M")]:
    if agg(t, 240) and agg(r, 240) and t in meta and r in meta:
        S.append("  %-6s %5.2f tok/s val %.3f  vs  %-12s %5.2f tok/s val %.3f  "
                 "(%+.1f%% speed, %+.3f val)" % (
                     t, agg(t, 240), meta[t]["val"], r, agg(r, 240), meta[r]["val"],
                     100 * (agg(r, 240) - agg(t, 240)) / agg(t, 240),
                     meta[r]["val"] - meta[t]["val"]))
txt = "\n".join(S)
open(O("summary.txt"), "w").write(txt)
print("\n" + txt)
print("\nwrote figures + tables.tex + summary.txt to ./%s/" % A.out)
