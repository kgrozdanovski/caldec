// multi_bench2: multi_bench with inputs found by name (any model): an input whose <tablesdir>/<name>.bin exists is
// written once (tables); every other input is read per item from <casedir>/<j>_<name>.bin. Used for the packed model.
// Deployment-style benchmark: load every bucket model into one process, then answer the questions in their original
// order, each on the smallest bucket that fits. Times inputs_set + rknn_run + outputs_get (end to end, input files
// are pre-loaded). Prints memory after each load.
// Build on the board: gcc -O2 -o multi_bench multi_bench.c -I. -L. -lrknnrt -Wl,-rpath,'$ORIGIN'
// Usage: ./multi_bench <order.txt> <out_logits.bin> <rounds>
//   order.txt line 1: "<nbuckets>"; then per bucket: "<S> <model.rknn> <casedir> <tablesdir|-> <share 0|1>" (share: init with
//   RKNN_FLAG_SHARE_WEIGHT_MEM from bucket 0, for remove_weight models); then per question: "<bucket_idx> <local_idx>"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

#define MAXB 16
#define NIN 16
#include <unistd.h>
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void *slurp(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { perror("read"); exit(2); } fclose(f); return b;
}
static void meminfo(const char *tag) {
  FILE *f = fopen("/proc/meminfo", "r"); char k[64]; long v; char u[8]; long avail = -1, shmem = -1;
  while (fscanf(f, "%63s %ld %7s", k, &v, u) == 3) {
    if (!strcmp(k, "MemAvailable:")) avail = v;
    if (!strcmp(k, "Shmem:")) shmem = v;
  }
  fclose(f); printf("mem %-16s MemAvailable %ld MB  Shmem %ld MB\n", tag, avail / 1024, shmem / 1024); fflush(stdout);
}
static void put(rknn_context c, rknn_tensor_mem *m, const void *src, size_t srcsz, rknn_tensor_type t) {
  if (t == RKNN_TENSOR_FLOAT16) { const float *f = src; __fp16 *h = m->virt_addr; size_t n = srcsz / 4; for (size_t i = 0; i < n; i++) h[i] = (__fp16)f[i]; }
  else memcpy(m->virt_addr, src, srcsz);
  rknn_mem_sync(c, m, RKNN_MEMORY_SYNC_TO_DEVICE);
}
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s order.txt out.bin rounds\n", argv[0]); return 1; }
  FILE *of = fopen(argv[1], "r"); if (!of) { perror(argv[1]); return 1; }
  int nb; if (fscanf(of, "%d", &nb) != 1 || nb > MAXB) return 1;
  int ZC = getenv("ZC") && atoi(getenv("ZC")); rknn_tensor_mem *zm[MAXB][NIN]; rknn_tensor_attr na[MAXB][NIN];
  int S[MAXB], share[MAXB], nin[MAXB]; char model[MAXB][256], cdir[MAXB][256], tdir[MAXB][256]; rknn_context ctx[MAXB]; rknn_tensor_attr ia[MAXB][NIN];
  void *tab[MAXB][NIN]; size_t tabsz[MAXB][NIN];
  meminfo("start");
  for (int b = 0; b < nb; b++) {
    if (fscanf(of, "%d %255s %255s %255s %d", &S[b], model[b], cdir[b], tdir[b], &share[b]) != 5) return 1;
    size_t n; void *m = slurp(model[b], &n);
    double t0 = now_ms();
    rknn_init_extend ext; memset(&ext, 0, sizeof ext); ext.ctx = ctx[0];
    if (rknn_init(&ctx[b], m, n, share[b] ? RKNN_FLAG_SHARE_WEIGHT_MEM : 0, share[b] ? &ext : NULL)) { fprintf(stderr, "rknn_init %s failed\n", model[b]); return 3; }
    if (!share[b]) free(m);   /* keep the buffer of shared models alive (conservative) */
    rknn_input_output_num io; rknn_query(ctx[b], RKNN_QUERY_IN_OUT_NUM, &io, sizeof io); nin[b] = io.n_input;
    if (nin[b] > NIN) { fprintf(stderr, "too many inputs\n"); return 3; }
    for (int k = 0; k < nin[b]; k++) { memset(&ia[b][k], 0, sizeof(ia[b][k])); ia[b][k].index = k; rknn_query(ctx[b], RKNN_QUERY_INPUT_ATTR, &ia[b][k], sizeof(ia[b][k]));
      char p[512]; snprintf(p, sizeof p, "%s/%s.bin", tdir[b], ia[b][k].name); tab[b][k] = NULL;
      if (strcmp(tdir[b], "-") && access(p, R_OK) == 0) tab[b][k] = slurp(p, &tabsz[b][k]); }
    if (ZC) for (int k = 0; k < nin[b]; k++) {   /* zero-copy: native-format input buffers bound once */
      memset(&na[b][k], 0, sizeof na[b][k]); na[b][k].index = k; rknn_query(ctx[b], RKNN_QUERY_NATIVE_INPUT_ATTR, &na[b][k], sizeof na[b][k]);
      zm[b][k] = rknn_create_mem(ctx[b], na[b][k].size_with_stride);
      if (!zm[b][k] || rknn_set_io_mem(ctx[b], zm[b][k], &na[b][k])) { fprintf(stderr, "set_io_mem failed b=%d k=%d\n", b, k); return 7; }
      if (tab[b][k]) put(ctx[b], zm[b][k], tab[b][k], tabsz[b][k], na[b][k].type);   /* tables: written once */
    }
    char tag[32]; snprintf(tag, sizeof tag, "after S=%d", S[b]); printf("init S=%d %.0f ms\n", S[b], now_ms() - t0); meminfo(tag);
  }
  int qb[2048], ql[2048], nq = 0;
  while (nq < 2048 && fscanf(of, "%d %d", &qb[nq], &ql[nq]) == 2) nq++;
  fclose(of);
  // pre-load all inputs
  static void *buf[2048][NIN]; static size_t sz[2048][NIN];
  for (int q = 0; q < nq; q++) for (int k = 0; k < nin[qb[q]]; k++) {
    if (tab[qb[q]][k]) continue;
    char p[512]; snprintf(p, sizeof p, "%s/%d_%s.bin", cdir[qb[q]], ql[q], ia[qb[q]][k].name); buf[q][k] = slurp(p, &sz[q][k]);
  }
  int rounds = atoi(argv[3]); double *lat = calloc(nq * rounds, sizeof(double)); int nl = 0;
  double *blat[MAXB]; int nbl[MAXB] = {0}; for (int b = 0; b < nb; b++) blat[b] = calloc(nq * rounds, sizeof(double));
  FILE *out = fopen(argv[2], "wb"); if (!out) { perror(argv[2]); return 2; }
  // warm-up: one run per bucket
  for (int r = -1; r < rounds; r++) {
    for (int q = 0; q < nq; q++) {
      int b = qb[q];
      if (r < 0) { static int warm[MAXB]; if (warm[b]) continue; warm[b] = 1; }
      rknn_input in[NIN]; memset(in, 0, sizeof(in));
      for (int k = 0; k < nin[b]; k++) {
        const char *nm = ia[b][k].name; int is_int = !strcmp(nm, "input_ids") || !strcmp(nm, "qtype");
        in[k].index = k; in[k].pass_through = 0;
        if (!tab[b][k]) { in[k].buf = buf[q][k]; in[k].size = sz[q][k]; } else { in[k].buf = tab[b][k]; in[k].size = tabsz[b][k]; }
        in[k].type = is_int ? RKNN_TENSOR_INT64 : RKNN_TENSOR_FLOAT32; in[k].fmt = RKNN_TENSOR_UNDEFINED;
      }
      double t0 = now_ms();
      if (ZC) { for (int k = 0; k < nin[b]; k++) if (!tab[b][k]) put(ctx[b], zm[b][k], buf[q][k], sz[q][k], na[b][k].type); }
      else if (rknn_inputs_set(ctx[b], nin[b], in)) { fprintf(stderr, "inputs_set failed\n"); return 4; }
      if (rknn_run(ctx[b], NULL)) { fprintf(stderr, "run failed\n"); return 5; }
      rknn_output o[1]; memset(o, 0, sizeof(o)); o[0].want_float = 1;
      if (rknn_outputs_get(ctx[b], 1, o, NULL)) { fprintf(stderr, "outputs_get failed\n"); return 6; }
      double dt = now_ms() - t0;
      if (r == 0) fwrite(o[0].buf, 1, o[0].size, out);
      rknn_outputs_release(ctx[b], 1, o);
      if (r >= 0) { lat[nl++] = dt; blat[b][nbl[b]++] = dt; }
    }
  }
  fclose(out);
  double sum = 0; for (int i = 0; i < nl; i++) sum += lat[i];
  qsort(lat, nl, sizeof(double), cmpd);
  printf("items %d x %d rounds: e2e mean %.1f ms  median %.1f  p95 %.1f  max %.1f\n", nq, rounds, sum / nl, lat[nl / 2],
         lat[(int)(0.95 * (nl - 1) + 0.5)], lat[nl - 1]);
  for (int b = 0; b < nb; b++) if (nbl[b]) {
    qsort(blat[b], nbl[b], sizeof(double), cmpd); printf("  S=%3d: %4d runs, e2e p50 %.1f ms\n", S[b], nbl[b], blat[b][nbl[b] / 2]);
  }
  meminfo("end");
  for (int b = 0; b < nb; b++) rknn_destroy(ctx[b]);
  return 0;
}
