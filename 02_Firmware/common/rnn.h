// Portable single-header inference for the recurrent baselines (LSTM / GRU).
// Same code runs on the host (verify against the PyTorch golden) and on the
// ESP32-S3 (mmap'd flash). Dims come from the model.bin header.
//
// Mirrors src/train_rnn.py:RecurrentLM.step op-for-op.
//   LSTM: g = W_ih x + b_ih + W_hh h + b_hh
//         i,f,g,o = chunk(g,4);  c = sig(f)*c + sig(i)*tanh(g);  h = sig(o)*tanh(c)
//   GRU:  gx = W_ih x + b_ih;   gh = W_hh h + b_hh
//         r = sig(gx0+gh0); z = sig(gx1+gh1); n = tanh(gx2 + r*gh2)
//         h = (1-z)*n + z*h
// Head: logits = tok_emb . RMSNorm(out_proj h_last)   (tied embedding)
//
// Quantised tensors use the identical group-128 ragged int4 + fp16-scale
// format as firmware/common/llm.h, so the packing code is shared verbatim.
#ifndef RNN_H
#define RNN_H
#include <stdint.h>
#include <math.h>
#include <string.h>

#define RNN_MAGIC 0x524E4E31u   // 'RNN1'
#define RNN_RMS_EPS 1e-6f

typedef struct {
  int vocab, dim, hidden, n_layers, cell, group;   // cell: 0 = LSTM, 1 = GRU
} RCfg;

// Group-wise int4 tensor viewed in place: ragged packed nibbles (row-aligned to
// a byte) + fp16 group scales. Identical layout to llm.h.
typedef struct {
  const uint8_t  *codes;
  const uint16_t *scales;
  int rows, cols, group, n_groups, row_bytes;
} RQT;

static inline float rnn_half2float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
  if (exp == 0) {
    if (man == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3FF; f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; memcpy(&out, &f, 4); return out;
}

static const uint8_t *rnn_bind_q(const uint8_t *p, RQT *t, int rows, int cols) {
  int32_t group; memcpy(&group, p, 4); p += 4;
  t->rows = rows; t->cols = cols; t->group = group;
  t->n_groups = (cols + group - 1) / group;
  t->row_bytes = (cols + 1) / 2;
  t->codes = p;  p += (size_t)rows * t->row_bytes;
  t->scales = (const uint16_t *)p;  p += (size_t)rows * t->n_groups * 2;
  return p;
}
static const uint8_t *rnn_bind_f(const uint8_t *p, const float **t, int n) {
  *t = (const float *)p;  return p + (size_t)n * sizeof(float);
}

// Dequantize row r into out[cols]. Used for the embedding lookup.
static inline void rnn_deq_row(const RQT *t, int r, float *out) {
  const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
  const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
  for (int gi = 0; gi < t->n_groups; gi++) {
    int begin = gi * t->group, end = begin + t->group;
    if (end > t->cols) end = t->cols;
    float scale = rnn_half2float(sc[gi]);
    for (int j = begin; j < end; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      out[j] = (float)(code - 8) * scale;
    }
  }
}

// y[row_begin:row_end] = W x. Ranged so a platform can split the output head
// across cores without touching any individual dot product.
static inline void rnn_matvec_range(const RQT *t, const float *x, float *y,
                                    int row_begin, int row_end) {
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float acc = 0.f;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group, end = begin + t->group;
      if (end > t->cols) end = t->cols;
      float scale = rnn_half2float(sc[gi]);
      float g = 0.f;
      int j = begin;
      if ((j & 1) && j < end) { g += (float)((row[j >> 1] >> 4) - 8) * x[j]; j++; }
      for (; j + 1 < end; j += 2) {
        uint8_t byte = row[j >> 1];
        g += (float)((byte & 0xF) - 8) * x[j];
        g += (float)((byte >> 4) - 8) * x[j + 1];
      }
      if (j < end) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        g += (float)(code - 8) * x[j];
      }
      acc += g * scale;
    }
    y[r] = acc;
  }
}
static inline void rnn_matvec(const RQT *t, const float *x, float *y) {
  rnn_matvec_range(t, x, y, 0, t->rows);
}

static inline void rnn_rmsnorm(const float *x, const float *w, int n, float *out) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / n + RNN_RMS_EPS);
  for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}
static inline float rnn_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

#define RNN_MAX_LAYERS 16

typedef struct {
  RCfg c;
  int gates;                       // 4 for LSTM, 3 for GRU
  RQT tok_emb;                     // [V, D]  (tied output head)
  RQT w_ih[RNN_MAX_LAYERS];        // [G*H, D or H]
  RQT w_hh[RNN_MAX_LAYERS];        // [G*H, H]
  const float *b_ih[RNN_MAX_LAYERS];
  const float *b_hh[RNN_MAX_LAYERS];
  RQT out_proj;                    // [D, H]
  const float *out_norm;           // [D]
  void (*head_matvec)(const RQT *, const float *, float *);  // platform override
} RModel;

// Parse header + bind every tensor in place. 0 on success, -1 on bad magic.
static int rnn_load(const uint8_t *base, RModel *m) {
  const uint8_t *p = base;
  uint32_t magic; memcpy(&magic, p, 4); p += 4;
  if (magic != RNN_MAGIC) return -1;
  int32_t hv[8]; memcpy(hv, p, 32); p += 32;
  m->c.vocab = hv[0]; m->c.dim = hv[1]; m->c.hidden = hv[2];
  m->c.n_layers = hv[3]; m->c.cell = hv[4]; m->c.group = hv[5];
  m->head_matvec = NULL;
  if (m->c.n_layers > RNN_MAX_LAYERS) return -2;
  int V = m->c.vocab, D = m->c.dim, H = m->c.hidden, L = m->c.n_layers;
  int G = m->gates = (m->c.cell == 0) ? 4 : 3;

  p = rnn_bind_q(p, &m->tok_emb, V, D);
  for (int l = 0; l < L; l++) {
    p = rnn_bind_q(p, &m->w_ih[l], G * H, l == 0 ? D : H);
    p = rnn_bind_q(p, &m->w_hh[l], G * H, H);
    p = rnn_bind_f(p, &m->b_ih[l], G * H);
    p = rnn_bind_f(p, &m->b_hh[l], G * H);
  }
  p = rnn_bind_q(p, &m->out_proj, D, H);
  p = rnn_bind_f(p, &m->out_norm, D);
  return 0;
}

// Caller-allocated scratch. Unlike the transformer there is no KV cache: the
// recurrent state is L*H (GRU) or 2*L*H (LSTM) floats, constant in position.
typedef struct {
  float *x;        // [max(D, H)]
  float *h;        // [L*H]
  float *c;        // [L*H]   (LSTM only; may be NULL for GRU)
  float *gx;       // [G*H]
  float *gh;       // [G*H]
  float *y;        // [D]
  float *logits;   // [V]
#ifdef LLM_PROFILE
  struct {
    uint64_t input_us, attn_us, ffn_us, ple_us, head_us;
    uint32_t calls;
  } profile;
#endif
} RScratch;

#ifdef LLM_PROFILE
static void rnn_profile_reset(RScratch *s) { memset(&s->profile, 0, sizeof(s->profile)); }
#endif

// Zero the recurrent state. Call once before a generation.
static void rnn_reset(const RModel *m, RScratch *s) {
  int n = m->c.n_layers * m->c.hidden;
  memset(s->h, 0, n * sizeof(float));
  if (s->c) memset(s->c, 0, n * sizeof(float));
}

// One decode step: token -> logits[V]. State persists across calls.
//
// Profiling buckets are named to match the transformer runtime so the same
// log parser works:  attn = input-path matvecs (W_ih), ffn = recurrent-path
// matvecs (W_hh), ple = output projection.
static void rnn_forward(RModel *m, int token, RScratch *s) {
  int D = m->c.dim, H = m->c.hidden, L = m->c.n_layers, G = m->gates;
#ifdef LLM_PROFILE
  uint64_t prev = (uint64_t)LLM_PROFILE_NOW(), now;
#endif

  rnn_deq_row(&m->tok_emb, token, s->x);
#ifdef LLM_PROFILE
  now = (uint64_t)LLM_PROFILE_NOW();
  s->profile.input_us += now - prev;
  prev = now;
#endif

  const float *in = s->x;
  for (int l = 0; l < L; l++) {
    float *hl = s->h + (size_t)l * H;
    rnn_matvec(&m->w_ih[l], in, s->gx);
    for (int i = 0; i < G * H; i++) s->gx[i] += m->b_ih[l][i];
#ifdef LLM_PROFILE
    now = (uint64_t)LLM_PROFILE_NOW();
    s->profile.attn_us += now - prev;
    prev = now;
#endif
    rnn_matvec(&m->w_hh[l], hl, s->gh);
    for (int i = 0; i < G * H; i++) s->gh[i] += m->b_hh[l][i];

    if (m->c.cell == 0) {                       // LSTM
      float *cl = s->c + (size_t)l * H;
      for (int i = 0; i < H; i++) {
        float gi = rnn_sigmoid(s->gx[i]           + s->gh[i]);
        float gf = rnn_sigmoid(s->gx[H + i]       + s->gh[H + i]);
        float gg = tanhf      (s->gx[2 * H + i]   + s->gh[2 * H + i]);
        float go = rnn_sigmoid(s->gx[3 * H + i]   + s->gh[3 * H + i]);
        cl[i] = gf * cl[i] + gi * gg;
        hl[i] = go * tanhf(cl[i]);
      }
    } else {                                    // GRU
      for (int i = 0; i < H; i++) {
        float r = rnn_sigmoid(s->gx[i]         + s->gh[i]);
        float z = rnn_sigmoid(s->gx[H + i]     + s->gh[H + i]);
        float n = tanhf(s->gx[2 * H + i] + r * s->gh[2 * H + i]);
        hl[i] = (1.f - z) * n + z * hl[i];
      }
    }
    in = hl;
#ifdef LLM_PROFILE
    now = (uint64_t)LLM_PROFILE_NOW();
    s->profile.ffn_us += now - prev;
    prev = now;
#endif
  }

  rnn_matvec(&m->out_proj, in, s->y);
  rnn_rmsnorm(s->y, m->out_norm, D, s->y);
#ifdef LLM_PROFILE
  now = (uint64_t)LLM_PROFILE_NOW();
  s->profile.ple_us += now - prev;
  prev = now;
#endif

  if (m->head_matvec) m->head_matvec(&m->tok_emb, s->y, s->logits);
  else rnn_matvec(&m->tok_emb, s->y, s->logits);
#ifdef LLM_PROFILE
  s->profile.head_us += (uint64_t)LLM_PROFILE_NOW() - prev;
  s->profile.calls++;
#endif
}

#endif
