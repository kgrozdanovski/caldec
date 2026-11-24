// Run an RKNN CalDec model on the NPU over the exported test cases; write logits; report latency.
// Build on the device: gcc -O2 -o rknn_caldec_bench rknn_caldec_bench.c -I. -L. -lrknnrt -Wl,-rpath,'$ORIGIN'
// Usage: ./rknn_caldec_bench model.rknn cases_dir n_cases runs out_logits.bin [perf]
// cases_dir/<i>_<name>.bin: input_ids/attention_mask/marker_pos/qtype as int64, marker_mask as float32.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include "rknn_api.h"

static void *slurp(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { perror("read"); exit(2); } fclose(f); return b;
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static double cpu_ms(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void ru(double *u, double *sy) { struct rusage r; getrusage(RUSAGE_SELF, &r); *u = r.ru_utime.tv_sec * 1e3 + r.ru_utime.tv_usec / 1e3; *sy = r.ru_stime.tv_sec * 1e3 + r.ru_stime.tv_usec / 1e3; }
static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s model.rknn cases_dir n runs out.bin [perf]\n", argv[0]); return 1; }
  int n = atoi(argv[3]), runs = atoi(argv[4]), perf = argc > 6;
  size_t msz; void *model = slurp(argv[1], &msz);
  rknn_context ctx; uint32_t flag = perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
  double t0 = now_ms();
  int ret = rknn_init(&ctx, model, msz, flag, NULL);
  if (ret) { fprintf(stderr, "rknn_init failed %d\n", ret); return 3; }
  printf("init_ms %.0f\n", now_ms() - t0);
  rknn_sdk_version ver; rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
  printf("sdk %s driver %s\n", ver.api_version, ver.drv_version);
  rknn_input_output_num io; rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  const char *names[5] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
  rknn_tensor_attr ia[16];
  for (uint32_t i = 0; i < io.n_input; i++) {
    memset(&ia[i], 0, sizeof(ia[i])); ia[i].index = i; rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[i], sizeof(ia[i]));
    printf("in%u %s type=%s fmt=%s n_dims=%u n_elems=%u\n", i, ia[i].name, get_type_string(ia[i].type), get_format_string(ia[i].fmt), ia[i].n_dims, ia[i].n_elems);
  }
  FILE *out = fopen(argv[5], "wb"); if (!out) { perror(argv[5]); return 2; }
  double *e2e = calloc(runs > n ? runs : n, sizeof(double)); int ne = 0; double cpu_tot = 0, usr_tot = 0, sys_tot = 0; double *lat = calloc(runs > n ? runs : n, sizeof(double)); int nl = 0;
  for (int it = 0; it < (runs > n ? runs : n); it++) {
    int c = it % n; int ni = io.n_input; rknn_input in[16]; void *bufs[16]; memset(in, 0, sizeof(in));
    for (int k = 0; k < ni; k++) {
      char p[512]; size_t sz; snprintf(p, sizeof p, "%s/%d_%s.bin", argv[2], c, ia[k].name[0] ? ia[k].name : names[k]);
      bufs[k] = slurp(p, &sz);
      in[k].index = k; in[k].buf = bufs[k]; in[k].size = sz; in[k].pass_through = 0;
      const char *nm = ia[k].name; int is_int = !strcmp(nm, "input_ids") || !strcmp(nm, "attention_mask") || !strcmp(nm, "marker_pos") || !strcmp(nm, "qtype");
      in[k].type = is_int ? RKNN_TENSOR_INT64 : RKNN_TENSOR_FLOAT32;
      in[k].fmt = ia[k].n_dims == 4 ? RKNN_TENSOR_NCHW : RKNN_TENSOR_UNDEFINED;   /* masks are [1,1,S,S] NCHW */
    }
    double t_set0 = now_ms();
    if ((ret = rknn_inputs_set(ctx, ni, in))) { fprintf(stderr, "inputs_set %d\n", ret); return 4; }
    double s = now_ms(), cs = cpu_ms(), u0, s0; ru(&u0, &s0);
    if ((ret = rknn_run(ctx, NULL))) { fprintf(stderr, "run %d\n", ret); return 5; }
    double e = now_ms(), ce = cpu_ms(), u1, s1; ru(&u1, &s1); cpu_tot += ce - cs; usr_tot += u1 - u0; sys_tot += s1 - s0;
    int no = io.n_output > 16 ? 16 : io.n_output; rknn_output o[16]; memset(o, 0, sizeof(o));
    for (int q = 0; q < no; q++) o[q].want_float = 1;
    if ((ret = rknn_outputs_get(ctx, no, o, NULL))) { fprintf(stderr, "outputs_get %d\n", ret); return 6; }
    if (it < n) for (int q = 0; q < no; q++) { fwrite(o[q].buf, 1, o[q].size, out); if (it == 0) printf("out%d bytes=%u\n", q, o[q].size); }
    rknn_outputs_release(ctx, no, o);
    if (it >= 2) e2e[ne++] = now_ms() - t_set0;
    for (int k = 0; k < ni; k++) free(bufs[k]);
    if (it >= 2) lat[nl++] = e - s;   /* first two runs are warm-up */
  }
  fclose(out);
  qsort(lat, nl, sizeof(double), cmp);
  { int nr = runs > n ? runs : n; printf("cpu_ms_per_run_in_rknn_run %.1f (user %.1f sys %.1f)\n", cpu_tot / nr, usr_tot / nr, sys_tot / nr); }
  qsort(e2e, ne, sizeof(double), cmp); if (ne) printf("e2e_ms (inputs_set+run+outputs_get) p50 %.1f p95 %.1f\n", e2e[ne / 2], e2e[(int)(0.95 * (ne - 1) + 0.5)]);
  printf("runs %d p50_ms %.1f p95_ms %.1f min_ms %.1f\n", nl, lat[nl / 2], lat[(int)(0.95 * (nl - 1) + 0.5)], lat[0]);
  if (perf) {
    rknn_perf_detail pd; if (!rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof(pd))) {
      FILE *f = fopen("perf_detail.txt", "w"); fputs(pd.perf_data, f); fclose(f); printf("perf detail -> perf_detail.txt\n"); }
    rknn_perf_run pr; if (!rknn_query(ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof(pr))) printf("perf_run_us %lld\n", (long long)pr.run_duration);
  }
  rknn_destroy(ctx); return 0;
}
