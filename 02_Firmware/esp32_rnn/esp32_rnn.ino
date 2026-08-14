// LSTM / GRU inference on the ESP32-S3, instrumented to match the PLE
// transformer sketch exactly so the two are directly comparable.
//
// Same measurement protocol: model mmap'd from the `model` flash partition at
// 0x110000, tied output head unpacked from int4 to int8 in PSRAM once at boot
// and split across both LX7 cores, greedy decoding, identical boot banner and
// per-stage profile format.
//
// The profile buckets are named for the transformer's stages so the same log
// parser works unmodified:
//     input -> embedding lookup
//     attn  -> input-path matvecs   (W_ih x)
//     ffn   -> recurrent-path matvecs (W_hh h) and the cell nonlinearity
//     ple   -> output projection + RMSNorm
//     head  -> tied output head
//
// Note on decoded text: vocab.h is the 25,353-entry TinyStories blob from the
// upstream project, while these models use the 4,096-token BPE. Emitted
// characters are therefore not the model's actual output. Serial writing is
// retained because the transformer sketch does it, and dropping it would make
// the throughput comparison unfair. Timing is unaffected.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/rnn.h"
#include "vocab.h"

#ifndef N_GEN
#define N_GEN 200
#endif
#ifndef WDT_EVERY
#define WDT_EVERY 8
#endif
#define FW_STR2(x) #x
#define FW_STR(x) FW_STR2(x)
#define FW_TAG   "rnn-v1-n" FW_STR(N_GEN) "-w" FW_STR(WDT_EVERY)
#define MARK_PIN 21

static const int PROMPT_IDS[] = {433, 447, 259, 405};
static const int N_GENERATE = N_GEN;

RModel model;
RScratch s;

// ---- allocators -------------------------------------------------------------
static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}
static void *ps_internal(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) { Serial.printf("SRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

// ---- int8 output head, staged once in PSRAM, split across both cores --------
// Identical strategy to the transformer sketch: the head is scanned in full on
// every token and dominates otherwise, so nibbles are unpacked once at boot.
static int8_t *head_w8 = NULL;
static float  *head_scale8 = NULL;
static int head_rows, head_cols;
static int8_t *head_actq = NULL;
static float  head_acts;

static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

static void head_rows_range(float *y, int r0, int r1) {
  for (int r = r0; r < r1; r++)
    y[r] = (float)dot_i8(head_actq, head_w8 + (size_t)r * head_cols, head_cols)
           * head_scale8[r] * head_acts;
}

static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

static void quantize_act_local(const float *x, int n, int8_t *xq, float *scale) {
  float xmax = 1e-8f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > xmax) xmax = a; }
  float inv = 127.f / xmax;
  for (int j = 0; j < n; j++) {
    int q = (int)lrintf(x[j] * inv);
    xq[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
  *scale = xmax / 127.f;
}

static void head_matvec_int8(const RQT *t, const float *x, float *y) {
  (void)t;
  quantize_act_local(x, head_cols, head_actq, &head_acts);
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void stage_head_int8(RQT *t) {
  head_rows = t->rows; head_cols = t->cols;
  head_w8 = (int8_t *)ps((size_t)head_rows * head_cols);
  head_scale8 = (float *)ps((size_t)head_rows * sizeof(float));
  head_actq = (int8_t *)ps_internal(head_cols);
  for (int r = 0; r < head_rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = head_w8 + (size_t)r * head_cols;
    for (int j = 0; j < head_cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    head_scale8[r] = rnn_half2float(t->scales[(size_t)r * t->n_groups]);
  }
  Serial.printf("head staged int8: %.2f MB\n",
                ((size_t)head_rows * head_cols + (size_t)head_rows * 4) / 1e6);
}

static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 Recurrent LM ===");
  pinMode(MARK_PIN, OUTPUT);
  digitalWrite(MARK_PIN, LOW);
  Serial.printf("fw: %s\n", FW_TAG);
  Serial.printf("cpu_mhz: %u\n", (unsigned)getCpuFrequencyMhz());
  Serial.printf("flash_hz: %u  flash_mode: %d\n",
                (unsigned)ESP.getFlashChipSpeed(), (int)ESP.getFlashChipMode());
  Serial.printf("temp_start: %.1f\n", temperatureRead());

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &h) != ESP_OK) {
    Serial.println("mmap failed"); return;
  }

  int rc = rnn_load((const uint8_t *)base, &model);
  if (rc) { Serial.printf("bad model (%d)\n", rc); return; }
  RCfg *c = &model.c;
  int V = c->vocab, D = c->dim, H = c->hidden, L = c->n_layers, G = model.gates;
  Serial.printf("model: cell=%s V=%d D=%d H=%d L=%d  (mapped %.1f MB)\n",
                c->cell == 0 ? "LSTM" : "GRU", V, D, H, L, part->size / 1e6);

  stage_head_int8(&model.tok_emb);
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                              &head_worker, 0) != pdPASS) {
    Serial.println("head worker creation failed"); return;
  }
  model.head_matvec = head_matvec_int8;

  int wide = D > H ? D : H;
  s.x      = (float *)ps((size_t)wide * 4);
  s.h      = (float *)ps((size_t)L * H * 4);
  s.c      = (c->cell == 0) ? (float *)ps((size_t)L * H * 4) : NULL;
  s.gx     = (float *)ps((size_t)G * H * 4);
  s.gh     = (float *)ps((size_t)G * H * 4);
  s.y      = (float *)ps((size_t)D * 4);
  s.logits = (float *)ps((size_t)V * 4);

  size_t state_bytes = (size_t)(c->cell == 0 ? 2 : 1) * L * H * 4;
  Serial.printf("recurrent state: %.1f KB (constant in position)\n", state_bytes / 1024.f);
  Serial.printf("PSRAM free after alloc: %u KB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

  // ---- generate ----
  rnn_reset(&model, &s);
  Serial.print(">>> ");
  int n_prompt = sizeof(PROMPT_IDS) / sizeof(int);
  int tok = 0;
  for (int i = 0; i < n_prompt; i++) {
    tok = PROMPT_IDS[i];
    emit(tok);
    rnn_forward(&model, tok, &s);
  }

  rnn_profile_reset(&s);
  digitalWrite(MARK_PIN, HIGH);
  int64_t t_start = esp_timer_get_time();
  int64_t decode_us = 0;
  int decoded = 0;

  for (int step = 0; step < N_GENERATE; step++) {
    int best = 0; float bv = -1e30f;
    int lim = V < VOCAB_N ? V : VOCAB_N;
    for (int v = 0; v < lim; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    tok = best;
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    rnn_forward(&model, tok, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & (WDT_EVERY - 1)) == 0) delay(0);
  }
  int64_t total_us = esp_timer_get_time() - t_start;
  digitalWrite(MARK_PIN, LOW);

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
  Serial.printf("temp_end: %.1f\n", temperatureRead());
  Serial.println("=== RUN COMPLETE ===");
  blink(0);
}

void loop() { delay(10000); }
