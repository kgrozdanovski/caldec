// Weight sharing test: base = full model (S=256); child = remove_weight model (S=128) initialised with
// RKNN_FLAG_SHARE_WEIGHT_MEM from base; ref = the same S=128 model with its own weights. Compares outputs bit-for-bit.
// Usage: ./share_test base.rknn child.rknn ref.rknn casedir tablesdir ncases
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"
static void *slurp(const char *p, size_t *n) { FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void *b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(2); fclose(f); return b; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static long shmem(void) { FILE *f = fopen("/proc/meminfo", "r"); char k[64], u[8]; long v, s = -1; while (fscanf(f, "%63s %ld %7s", k, &v, u) == 3) if (!strcmp(k, "Shmem:")) s = v; fclose(f); return s / 1024; }
static const char *N[9] = {"input_ids", "local_mask", "global_mask", "head_mask", "marker_sel", "marker_mask", "qtype", "rope_tab", "s_scale"};
static int run(rknn_context c, const char *cd, const char *td, int i, float *out6, double *ms) {
  rknn_input in[9]; void *b[9]; memset(in, 0, sizeof in);
  for (int k = 0; k < 9; k++) {
    char p[512]; size_t n;
    if (k < 7) snprintf(p, sizeof p, "%s/%d_%s.bin", cd, i, N[k]); else snprintf(p, sizeof p, "%s/%s.bin", td, N[k]);
    b[k] = slurp(p, &n); in[k].index = k; in[k].buf = b[k]; in[k].size = n;
    in[k].type = (k == 0 || k == 6) ? RKNN_TENSOR_INT64 : RKNN_TENSOR_FLOAT32; in[k].fmt = RKNN_TENSOR_UNDEFINED;
  }
  double t0 = now_ms();
  if (rknn_inputs_set(c, 9, in) || rknn_run(c, NULL)) return -1;
  rknn_output o[1]; memset(o, 0, sizeof o); o[0].want_float = 1; if (rknn_outputs_get(c, 1, o, NULL)) return -2;
  *ms = now_ms() - t0; memcpy(out6, o[0].buf, 6 * sizeof(float)); rknn_outputs_release(c, 1, o);
  for (int k = 0; k < 9; k++) free(b[k]); return 0;
}
int main(int argc, char **argv) {
  size_t n; void *m; rknn_context base, child, ref; int nc = atoi(argv[6]);
  printf("Shmem start %ld MB\n", shmem());
  m = slurp(argv[1], &n); if (rknn_init(&base, m, n, 0, NULL)) { puts("base init failed"); return 3; } printf("Shmem after base %ld MB\n", shmem());
  m = slurp(argv[2], &n); rknn_init_extend ext; memset(&ext, 0, sizeof ext); ext.ctx = base;
  int r = rknn_init(&child, m, n, RKNN_FLAG_SHARE_WEIGHT_MEM, &ext); printf("child init (share weights) ret %d; Shmem %ld MB\n", r, shmem()); if (r) return 4;
  m = slurp(argv[3], &n); if (rknn_init(&ref, m, n, 0, NULL)) { puts("ref init failed"); return 5; } printf("Shmem after ref %ld MB\n", shmem());
  int same = 0; double dmax = 0, tc = 0, tr = 0;
  for (int i = 0; i < nc; i++) {
    float a[6], b[6]; double ma, mb;
    if (run(child, argv[4], argv[5], i, a, &ma) || run(ref, argv[4], argv[5], i, b, &mb)) { puts("run failed"); return 6; }
    int eq = !memcmp(a, b, sizeof a); same += eq; tc += ma; tr += mb;
    for (int k = 0; k < 6; k++) { double d = a[k] - b[k]; if (d < 0) d = -d; if (b[k] > -5000 && d > dmax) dmax = d; }
  }
  printf("cases %d: bit-identical %d/%d, max |diff| %.6g; mean e2e child %.1f ms, ref %.1f ms\n", nc, same, nc, dmax, tc / nc, tr / nc);
  return 0;
}
