"""Recurrent baselines for the ESP32-S3 characterisation: LSTM and GRU.

Matched to the PLE transformer protocol exactly --- same tokenizer, same
train/val bins, same step count, batch size, sequence length, LR schedule and
evaluation --- so that validation loss is directly comparable. Only the
architecture differs.

Parameter accounting follows src/model.py: the tied embedding (vocab x d_model)
is reported as 'stream' and excluded from 'core'. Hidden size is solved to hit
a target core budget, mirroring the transformer's FFN solver.

Place next to src/ and run from the project root:

    python src/train_rnn.py --cell lstm --target-core 1500000 --n-layers 3 --tag c1.5M
    python src/train_rnn.py --cell gru  --target-core 1500000 --n-layers 3 --tag c1.5M
    python src/export_rnn.py lstm-c1.5M-s0
"""

import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RUNS = os.path.join(HERE, "..", "runs")

GATES = {"lstm": 4, "gru": 3}


# ----------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RecurrentLM(nn.Module):
    """Stacked LSTM or GRU with a tied input/output embedding.

    Written with explicit weight matrices rather than nn.LSTM so that the
    export and the C runtime can mirror the computation tensor for tensor.
    """

    def __init__(self, cell, vocab, d_model, hidden, n_layers):
        super().__init__()
        self.cell, self.V, self.D, self.H, self.L = cell, vocab, d_model, hidden, n_layers
        G = GATES[cell]
        self.emb = nn.Embedding(vocab, d_model)
        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()
        for l in range(n_layers):
            fan_in = d_model if l == 0 else hidden
            self.w_ih.append(nn.Parameter(torch.empty(G * hidden, fan_in)))
            self.w_hh.append(nn.Parameter(torch.empty(G * hidden, hidden)))
            self.b_ih.append(nn.Parameter(torch.zeros(G * hidden)))
            self.b_hh.append(nn.Parameter(torch.zeros(G * hidden)))
        self.out_proj = nn.Linear(hidden, d_model, bias=False)
        self.out_norm = RMSNorm(d_model)
        self._init()

    def _init(self):
        k = 1.0 / math.sqrt(self.H)
        for p in list(self.w_ih) + list(self.w_hh):
            nn.init.uniform_(p, -k, k)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.cell == "lstm":
            # forget-gate bias to 1.0 (standard; helps long-range retention)
            for b in self.b_ih:
                with torch.no_grad():
                    b[self.H:2 * self.H].fill_(1.0)

    def step(self, l, x, h, c):
        """One timestep of layer l. Mirrors the C runtime exactly."""
        gx = F.linear(x, self.w_ih[l], self.b_ih[l])
        gh = F.linear(h, self.w_hh[l], self.b_hh[l])
        H = self.H
        if self.cell == "lstm":
            g = gx + gh
            i = torch.sigmoid(g[..., 0:H])
            f = torch.sigmoid(g[..., H:2 * H])
            gg = torch.tanh(g[..., 2 * H:3 * H])
            o = torch.sigmoid(g[..., 3 * H:4 * H])
            c = f * c + i * gg
            return o * torch.tanh(c), c
        # GRU: the reset gate multiplies only the hidden contribution
        r = torch.sigmoid(gx[..., 0:H] + gh[..., 0:H])
        z = torch.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
        n = torch.tanh(gx[..., 2 * H:3 * H] + r * gh[..., 2 * H:3 * H])
        return (1 - z) * n + z * h, c

    def forward(self, idx, targets=None):
        B, T = idx.shape
        dev = idx.device
        x = self.emb(idx)                                  # [B,T,D]
        hs = [torch.zeros(B, self.H, device=dev) for _ in range(self.L)]
        cs = [torch.zeros(B, self.H, device=dev) for _ in range(self.L)]
        outs = []
        for t in range(T):
            inp = x[:, t]
            for l in range(self.L):
                hs[l], cs[l] = self.step(l, inp, hs[l], cs[l])
                inp = hs[l]
            outs.append(inp)
        y = torch.stack(outs, 1)                           # [B,T,H]
        y = self.out_norm(self.out_proj(y))                # [B,T,D]
        logits = F.linear(y, self.emb.weight)              # tied head
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, self.V), targets.reshape(-1))
        return logits, loss

    def param_counts(self):
        stream = self.V * self.D
        core = sum(p.numel() for n, p in self.named_parameters()
                   if not n.startswith("emb."))
        return dict(core=core, stream=stream, table=0, total=core + stream)


def core_params(cell, D, H, L):
    """Analytic core count, used by the hidden-size solver."""
    G = GATES[cell]
    n = G * H * D + G * H * H + 2 * G * H              # layer 0
    n += (L - 1) * (2 * G * H * H + 2 * G * H)         # layers 1..L-1
    n += D * H + D                                     # out_proj + out_norm
    return n


def solve_hidden(cell, D, L, target):
    lo, hi = 8, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        if core_params(cell, D, mid, L) < target:
            lo = mid + 1
        else:
            hi = mid
    # take whichever of lo-1, lo is closer
    a, b = lo - 1, lo
    return a if abs(core_params(cell, D, a, L) - target) < \
                abs(core_params(cell, D, b, L) - target) else b


# ----------------------------------------------------------------- data
class Batcher:
    def __init__(self, split, bs, sl, device, suffix=""):
        self.data = np.memmap(os.path.join(DATA, f"{split}{suffix}.bin"),
                              dtype=np.uint16, mode="r")
        self.bs, self.sl, self.device = bs, sl, device
        self.rng = np.random.default_rng(1234 if split == "val" else None)

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i:i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.sl] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))


@torch.no_grad()
def evaluate(model, batcher, iters):
    model.eval()
    batcher.rng = np.random.default_rng(1234)   # identical val batches every arm
    losses = [model(*batcher())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["lstm", "gru"], required=True)
    ap.add_argument("--target-core", type=int, default=1_500_000)
    ap.add_argument("--hidden", type=int, default=None, help="skip the solver")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = get_device()
    os.makedirs(RUNS, exist_ok=True)

    H = a.hidden or solve_hidden(a.cell, a.d_model, a.n_layers, a.target_core)
    model = RecurrentLM(a.cell, a.vocab, a.d_model, H, a.n_layers).to(dev)
    pc = model.param_counts()
    name = f"{a.cell}-{a.tag or 'h'+str(H)}-s{a.seed}"

    print(f"{name}: cell={a.cell} D={a.d_model} H={H} L={a.n_layers}")
    print(f"  core {pc['core']:,}  (target {a.target_core:,}, "
          f"off by {100*(pc['core']-a.target_core)/a.target_core:+.2f}%)")
    print(f"  stream {pc['stream']:,}   total {pc['total']:,}")

    suffix = "" if a.vocab == 4096 else f"_v{a.vocab}"
    tr = Batcher("train", a.batch_size, a.seq_len, dev, suffix)
    va = Batcher("val", a.batch_size, a.seq_len, dev, suffix)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    hist, best, t0 = [], float("inf"), time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        x, y = tr()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            v = evaluate(model, va, a.eval_iters)
            best = min(best, v)
            hist.append(dict(step=step, tokens=(step + 1) * a.batch_size * a.seq_len,
                             train=loss.item(), val=v))
            print(f"  step {step:5d}  train {loss.item():.4f}  val {v:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    final = hist[-1]["val"]
    torch.save(dict(cell=a.cell, cfg=dict(vocab=a.vocab, d_model=a.d_model,
                                          hidden=H, n_layers=a.n_layers),
                    state=model.state_dict()),
               os.path.join(RUNS, name + ".pt"))
    json.dump(dict(arm=a.cell, seed=a.seed, tag=a.tag,
                   config=dict(cell=a.cell, vocab_size=a.vocab,
                               d_model=a.d_model, hidden=H,
                               n_layers=a.n_layers, seq_len=a.seq_len),
                   params=pc, final_val=final, best_val=best,
                   final_ppl=math.exp(final),
                   tokens_seen=a.steps * a.batch_size * a.seq_len,
                   steps=a.steps, wall_seconds=time.time() - t0, history=hist),
              open(os.path.join(RUNS, name + ".json"), "w"), indent=1)
    print(f"\n{name}: val {final:.4f}  ppl {math.exp(final):.2f}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
