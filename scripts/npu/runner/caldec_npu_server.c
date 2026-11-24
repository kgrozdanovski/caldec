// caldec_npu_server: keeps every bucket model of a CalDec NPU set loaded (shared weights, zero-copy inputs, as in
// multi_bench2.c) and answers binary requests on stdin/stdout. The front-end (caldec_npu.py) builds the inputs.
//
// Build on the device: gcc -O2 -o caldec_npu_server caldec_npu_server.c -I. -L. -lrknnrt -Wl,-rpath,'$ORIGIN'
// Usage: ./caldec_npu_server <models.txt>
//   models.txt line 1: "<nbuckets>"; then per bucket "<S> <model.rknn> <tablesdir|-> <share 0|1>" (bucket 0 holds the
//   weights; share=1 models are remove_weight models initialised with RKNN_FLAG_SHARE_WEIGHT_MEM from bucket 0).
//   An input whose <tablesdir>/<name>.bin exists is written once; the others come with each request.
// Startup output (text, stdout): per bucket "B <S> <n_inputs> <name>:<i64|f32>:<elems>|T ... O<n_outputs>", then "READY".
// Request (binary, stdin):  int32 S, then for each non-table input in model order its raw data
//                           (int64 for input_ids, float32 otherwise; elems as announced).
// Reply (binary, stdout):   int32 status (0 ok), int32 n_outputs, then per output int32 n + float32[n] (output 0 =
//                           logits, output 1 = hcls if the model has it), then float32 ms (inputs + run + outputs).
// S = 0 ends the server.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "rknn_api.h"

#define MAXB 16
#define NIN 16
#define NOUT 4
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void *slurp(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { perror("read"); exit(2); } fclose(f); return b;
}
static void put(rknn_context c, rknn_tensor_mem *m, const void *src, size_t srcsz, rknn_tensor_type t) {
  if (t == RKNN_TENSOR_FLOAT16) { const float *f = src; __fp16 *h = m->virt_addr; size_t n = srcsz / 4; for (size_t i = 0; i < n; i++) h[i] = (__fp16)f[i]; }
  else memcpy(m->virt_addr, src, srcsz);
  rknn_mem_sync(c, m, RKNN_MEMORY_SYNC_TO_DEVICE);
}
static int is_int(const char *nm) { return !strcmp(nm, "input_ids") || !strcmp(nm, "qtype"); }
static int readn(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s models.txt\n", argv[0]); return 1; }
  FILE *of = fopen(argv[1], "r"); if (!of) { perror(argv[1]); return 1; }
  int nb; if (fscanf(of, "%d", &nb) != 1 || nb > MAXB) return 1;
  int S[MAXB], share[MAXB], nin[MAXB], nout[MAXB]; char model[MAXB][256], tdir[MAXB][256]; rknn_context ctx[MAXB];
  rknn_tensor_attr ia[MAXB][NIN], na[MAXB][NIN]; rknn_tensor_mem *zm[MAXB][NIN]; void *tab[MAXB][NIN]; size_t tabsz[MAXB][NIN];
  void *req[MAXB][NIN]; size_t reqsz[MAXB][NIN];
  for (int b = 0; b < nb; b++) {
    if (fscanf(of, "%d %255s %255s %d", &S[b], model[b], tdir[b], &share[b]) != 4) { fprintf(stderr, "bad models file\n"); return 1; }
    size_t n; void *m = slurp(model[b], &n);
    rknn_init_extend ext; memset(&ext, 0, sizeof ext); ext.ctx = ctx[0];
    if (rknn_init(&ctx[b], m, n, share[b] ? RKNN_FLAG_SHARE_WEIGHT_MEM : 0, share[b] ? &ext : NULL)) { fprintf(stderr, "rknn_init %s failed\n", model[b]); return 3; }
    if (!share[b]) free(m);
    rknn_input_output_num io; rknn_query(ctx[b], RKNN_QUERY_IN_OUT_NUM, &io, sizeof io); nin[b] = io.n_input; nout[b] = io.n_output;
    if (nin[b] > NIN || nout[b] > NOUT) { fprintf(stderr, "too many inputs/outputs\n"); return 3; }
    printf("B %d %d", S[b], nin[b]);
    for (int k = 0; k < nin[b]; k++) {
      memset(&ia[b][k], 0, sizeof ia[b][k]); ia[b][k].index = k; rknn_query(ctx[b], RKNN_QUERY_INPUT_ATTR, &ia[b][k], sizeof ia[b][k]);
      memset(&na[b][k], 0, sizeof na[b][k]); na[b][k].index = k; rknn_query(ctx[b], RKNN_QUERY_NATIVE_INPUT_ATTR, &na[b][k], sizeof na[b][k]);
      zm[b][k] = rknn_create_mem(ctx[b], na[b][k].size_with_stride);
      if (!zm[b][k] || rknn_set_io_mem(ctx[b], zm[b][k], &na[b][k])) { fprintf(stderr, "set_io_mem failed b=%d k=%d\n", b, k); return 7; }
      char p[512]; snprintf(p, sizeof p, "%s/%s.bin", tdir[b], ia[b][k].name); tab[b][k] = NULL; req[b][k] = NULL;
      if (strcmp(tdir[b], "-") && access(p, R_OK) == 0) {
        tab[b][k] = slurp(p, &tabsz[b][k]); put(ctx[b], zm[b][k], tab[b][k], tabsz[b][k], na[b][k].type);
        printf(" %s|T", ia[b][k].name);
      } else {
        reqsz[b][k] = (size_t)ia[b][k].n_elems * (is_int(ia[b][k].name) ? 8 : 4); req[b][k] = malloc(reqsz[b][k]);
        printf(" %s:%s:%u", ia[b][k].name, is_int(ia[b][k].name) ? "i64" : "f32", ia[b][k].n_elems);
      }
    }
    printf(" O%d\n", nout[b]); fflush(stdout);
  }
  fclose(of);
  printf("READY\n"); fflush(stdout);
  for (;;) {
    int32_t s; if (!readn(&s, 4) || s == 0) break;
    int b = -1; for (int i = 0; i < nb; i++) if (S[i] == s) b = i;
    if (b < 0) { fprintf(stderr, "unknown bucket %d\n", s); return 8; }
    for (int k = 0; k < nin[b]; k++) if (!tab[b][k] && !readn(req[b][k], reqsz[b][k])) { fprintf(stderr, "short request\n"); return 9; }
    double t0 = now_ms();
    for (int k = 0; k < nin[b]; k++) if (!tab[b][k]) put(ctx[b], zm[b][k], req[b][k], reqsz[b][k], na[b][k].type);
    int32_t st = rknn_run(ctx[b], NULL);
    rknn_output o[NOUT]; memset(o, 0, sizeof o); for (int k = 0; k < nout[b]; k++) o[k].want_float = 1;
    if (!st) st = rknn_outputs_get(ctx[b], nout[b], o, NULL);
    float ms = (float)(now_ms() - t0);
    int32_t no = st ? 0 : nout[b];
    fwrite(&st, 4, 1, stdout); fwrite(&no, 4, 1, stdout);
    for (int k = 0; k < no; k++) { int32_t n = (int32_t)(o[k].size / 4); fwrite(&n, 4, 1, stdout); fwrite(o[k].buf, 4, n, stdout); }
    fwrite(&ms, 4, 1, stdout); fflush(stdout);
    if (!st) rknn_outputs_release(ctx[b], nout[b], o);
  }
  for (int b = nb - 1; b >= 0; b--) rknn_destroy(ctx[b]);
  return 0;
}
