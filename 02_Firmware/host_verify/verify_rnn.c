// Host verification for the recurrent runtime: run the exported model through
// rnn.h and compare the last-position logits against the PyTorch golden.
//
//   cc -O3 -o verify_rnn verify_rnn.c -lm
//   ./verify_rnn model.bin golden.txt
//
// The golden logits come from the *dequantised* PyTorch model, so any
// disagreement here is a port bug, not quantisation error.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rnn.h"

static const int PROMPT[] = {1, 500, 1000, 200, 42, 777, 13, 99};
#define N_PROMPT (int)(sizeof(PROMPT) / sizeof(PROMPT[0]))

static uint8_t *slurp(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = (uint8_t *)malloc(*len);
  if (!buf || fread(buf, 1, *len, f) != *len) { fprintf(stderr, "read failed\n"); exit(1); }
  fclose(f);
  return buf;
}

static void *xmalloc(size_t n) {
  void *p = calloc(1, n);
  if (!p) { fprintf(stderr, "alloc failed (%zu bytes)\n", n); exit(1); }
  return p;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.bin golden.txt\n", argv[0]);
    return 1;
  }
  size_t nbytes;
  uint8_t *blob = slurp(argv[1], &nbytes);

  RModel m;
  int rc = rnn_load(blob, &m);
  if (rc) { fprintf(stderr, "rnn_load failed (%d) -- bad magic or too many layers\n", rc); return 1; }
  int V = m.c.vocab, D = m.c.dim, H = m.c.hidden, L = m.c.n_layers, G = m.gates;
  printf("loaded: %s V=%d D=%d H=%d L=%d group=%d  (%.2f MB)\n",
         m.c.cell == 0 ? "LSTM" : "GRU", V, D, H, L, m.c.group, nbytes / 1e6);

  RScratch s;
  int wide = D > H ? D : H;
  s.x      = (float *)xmalloc((size_t)wide * 4);
  s.h      = (float *)xmalloc((size_t)L * H * 4);
  s.c      = (m.c.cell == 0) ? (float *)xmalloc((size_t)L * H * 4) : NULL;
  s.gx     = (float *)xmalloc((size_t)G * H * 4);
  s.gh     = (float *)xmalloc((size_t)G * H * 4);
  s.y      = (float *)xmalloc((size_t)D * 4);
  s.logits = (float *)xmalloc((size_t)V * 4);

  printf("state: %.1f KB %s (transformer KV cache for L=6,S=512,D=128 is 3.15 MB)\n",
         (m.c.cell == 0 ? 2.f : 1.f) * L * H * 4 / 1024.f,
         m.c.cell == 0 ? "(h + c)" : "(h)");

  rnn_reset(&m, &s);
  for (int i = 0; i < N_PROMPT; i++) rnn_forward(&m, PROMPT[i], &s);

  // --- golden ---------------------------------------------------------------
  FILE *g = fopen(argv[2], "r");
  if (!g) { perror(argv[2]); return 1; }
  float *ref = (float *)xmalloc((size_t)V * 4);
  for (int i = 0; i < V; i++) {
    if (fscanf(g, "%f", &ref[i]) != 1) {
      fprintf(stderr, "golden has fewer than %d entries (stopped at %d)\n", V, i);
      return 1;
    }
  }
  fclose(g);

  const int probe[] = {265, 14, 1, 12, 13, 100, V - 1};
  printf("\nsample logits (idx: C vs ref):\n");
  for (unsigned i = 0; i < sizeof(probe) / sizeof(probe[0]); i++) {
    int k = probe[i];
    if (k < 0 || k >= V) continue;
    printf("  [%5d]  C=%9.4f  ref=%9.4f\n", k, s.logits[k], ref[k]);
  }

  double maxd = 0.0, sq = 0.0;
  int argmax_c = 0, argmax_r = 0;
  for (int i = 0; i < V; i++) {
    double d = fabs((double)s.logits[i] - (double)ref[i]);
    if (d > maxd) maxd = d;
    sq += d * d;
    if (s.logits[i] > s.logits[argmax_c]) argmax_c = i;
    if (ref[i] > ref[argmax_r]) argmax_r = i;
  }
  double rms = sqrt(sq / V);
  printf("\nmax abs diff = %.6f   rms diff = %.6f\n", maxd, rms);
  printf("argmax: C = %d, ref = %d  %s\n", argmax_c, argmax_r,
         argmax_c == argmax_r ? "(match)" : "(MISMATCH)");

  int ok = (maxd < 1e-3) && (argmax_c == argmax_r);
  printf("%s: C %s PyTorch golden\n", ok ? "PASS" : "FAIL",
         ok ? "matches" : "does NOT match");
  return ok ? 0 : 1;
}
