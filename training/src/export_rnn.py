"""Export a trained LSTM/GRU to the flat int4 binary the C runtime mmaps,
plus a golden logits reference so the C port can be proven correct on the host
before it touches hardware.

Packing is byte-identical to src/export.py: group-128 symmetric int4, ragged
(no padding), fp16 scales rounded *before* dequantisation, codes = value + 8,
even indices in the low nibble. The golden logits are the 4-bit model's
logits, so a C-vs-PyTorch comparison isolates port correctness from
quantisation error.

Format:
    magic 'RNN1'
    int32[8]  vocab, d_model, hidden, n_layers, cell(0=lstm,1=gru), group, 0, 0
    QT  tok_emb        [V, D]      (tied output head)
    for each layer l:
      QT  w_ih         [G*H, in]   in = D for l==0 else H
      QT  w_hh         [G*H, H]
      f32 b_ih         [G*H]
      f32 b_hh         [G*H]
    QT  out_proj       [D, H]
    f32 out_norm       [D]

    python src/export_rnn.py lstm-c1.5M-s0
"""

import os, struct, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rnn import RecurrentLM, GATES

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
OUT = os.path.join(HERE, "..", "firmware", "model")
MAGIC = 0x524E4E31          # 'RNN1'
GROUP = 128
PROMPT = [1, 500, 1000, 200, 42, 777, 13, 99]   # same as src/export.py


def quant_pack(w, group=GROUP):
    """Identical to src/export.py:quant_pack. Returns (packed, scales16, dq)."""
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    q = torch.zeros(rows, cols)
    dq = torch.zeros(rows, cols)
    scales = torch.zeros(rows, n_groups)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-8)
        sc = sc.half().float()
        scales[:, gi] = sc.squeeze(1)
        qi = torch.clamp(torch.round(seg / sc), -7, 7)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    dq = dq.reshape(out_shape)
    codes = (q.to(torch.int16) + 8).to(torch.uint8).numpy()
    row_bytes = (cols + 1) // 2
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    lo, hi = codes[:, 0::2], codes[:, 1::2]
    packed[:, : lo.shape[1]] = lo
    packed[:, : hi.shape[1]] |= (hi << 4)
    return packed.reshape(-1), scales.numpy().astype(np.float16).reshape(-1), dq


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "lstm-c1.5M-s0"
    os.makedirs(OUT, exist_ok=True)

    ck = torch.load(os.path.join(RUNS, tag + ".pt"), map_location="cpu",
                    weights_only=False)
    cfg, cell = ck["cfg"], ck["cell"]
    V, D, H, L = cfg["vocab"], cfg["d_model"], cfg["hidden"], cfg["n_layers"]
    G = GATES[cell]
    model = RecurrentLM(cell, V, D, H, L)
    model.load_state_dict(ck["state"])
    model.eval()
    sd = model.state_dict()

    # --- build the tensor plan in the exact order the C reader expects -------
    plan = []                                  # (name, tensor, quantize?)
    plan.append(("emb.weight", sd["emb.weight"], True))
    for l in range(L):
        plan.append((f"w_ih.{l}", sd[f"w_ih.{l}"], True))
        plan.append((f"w_hh.{l}", sd[f"w_hh.{l}"], True))
        plan.append((f"b_ih.{l}", sd[f"b_ih.{l}"], False))
        plan.append((f"b_hh.{l}", sd[f"b_hh.{l}"], False))
    plan.append(("out_proj.weight", sd["out_proj.weight"], True))
    plan.append(("out_norm.w", sd["out_norm.w"], False))

    # --- write ---------------------------------------------------------------
    path = os.path.join(OUT, "model.bin")
    dqs = {}
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<8i", V, D, H, L,
                            0 if cell == "lstm" else 1, GROUP, 0, 0))
        for name, t, quant in plan:
            if quant:
                packed, scales, dq = quant_pack(t)
                f.write(struct.pack("<i", GROUP))
                f.write(packed.tobytes())
                f.write(scales.tobytes())
                dqs[name] = dq
            else:
                f.write(t.detach().float().numpy().astype(np.float32).tobytes())
                dqs[name] = t.detach().float()
    size = os.path.getsize(path)
    print(f"wrote {path}  ({size/1e6:.2f} MB)  {len(plan)} tensors  "
          f"cell={cell} V={V} D={D} H={H} L={L}")

    # --- golden logits from the DEQUANTIZED model ---------------------------
    with torch.no_grad():
        for name, dq in dqs.items():
            key = name if name in sd else None
            if key is None:
                continue
            sd[key].copy_(dq.reshape(sd[key].shape))
        model.load_state_dict(sd)
        model.eval()
        idx = torch.tensor([PROMPT], dtype=torch.long)
        logits, _ = model(idx)
        last = logits[0, -1].float().numpy()

    gpath = os.path.join(OUT, "golden.txt")
    with open(gpath, "w") as f:
        for v in last:
            f.write("%.6f\n" % v)
    top5 = np.argsort(-last)[:5].tolist()
    print(f"golden: prompt={PROMPT}")
    print(f"golden: last-pos top5 token ids = {top5}")
    print(f"golden: logit range [{last.min():.3f}, {last.max():.3f}]")
    print(f"wrote {gpath}")

    # --- convenience copies for the benchmark pipeline -----------------------
    import shutil
    for suffix, src in ((".bin", path), ("_golden.txt", gpath)):
        dst = os.path.join(OUT, tag + suffix)
        shutil.copy2(src, dst)
    mj = os.path.join(RUNS, tag + ".json")
    if os.path.exists(mj):
        shutil.copy2(mj, os.path.join(OUT, tag + "_metrics.json"))
    print(f"copies: {tag}.bin, {tag}_golden.txt, {tag}_metrics.json")


if __name__ == "__main__":
    main()
