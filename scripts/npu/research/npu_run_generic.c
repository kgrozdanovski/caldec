// npu_run_generic: run ANY RKNN model (fp16 or int8) with float32 inputs in the ONNX layout (NCHW), through the plain
// rknn_inputs_set path, so the runtime converts type and layout itself. Same protocol as caldec_npu_server
// (models.txt with one model; tables "-"); used by npu_generic.py for research models (doc 26).
// Build: gcc -O2 -o npu_run_generic npu_run_generic.c -I/opt/caldec-npu -L/opt/caldec-npu -lrknnrt -Wl,-rpath,/opt/caldec-npu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"
#define NIN 64
#define NOUT 64
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
  if (argc < 2) return 1;
  FILE *of = fopen(argv[1], "r"); int nb, S, sh; char model[512], tdir[64];
  if (!of || fscanf(of, "%d %d %511s %63s %d", &nb, &S, model, tdir, &sh) != 5) { fprintf(stderr, "bad models file\n"); return 1; }
  FILE *f = fopen(model, "rb"); fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
  void *m = malloc(n); if (fread(m, 1, n, f) != n) return 2; fclose(f);
  rknn_context ctx; if (rknn_init(&ctx, m, n, 0, NULL)) { fprintf(stderr, "rknn_init failed\n"); return 3; }
  rknn_input_output_num io; rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io);
  if (io.n_input > NIN || io.n_output > NOUT) { fprintf(stderr, "too many inputs/outputs: %u/%u\n", io.n_input, io.n_output); return 3; }
  rknn_tensor_attr ia[NIN]; void *buf[NIN]; size_t bsz[NIN];
  printf("B %d %d", S, io.n_input);
  for (int k = 0; k < (int)io.n_input; k++) {
    memset(&ia[k], 0, sizeof ia[k]); ia[k].index = k; rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[k], sizeof ia[k]);
    bsz[k] = (size_t)ia[k].n_elems * 4; buf[k] = malloc(bsz[k]);
    printf(" %s:f32:%u", ia[k].name, ia[k].n_elems);
    fprintf(stderr, "input %d %s type %d fmt %d dims", k, ia[k].name, ia[k].type, ia[k].fmt);
    for (unsigned d = 0; d < ia[k].n_dims; d++) fprintf(stderr, " %u", ia[k].dims[d]); fprintf(stderr, "\n");
  }
  printf(" O%d\nREADY\n", io.n_output); fflush(stdout);
  int u8 = getenv("NPU_U8_NHWC") && atoi(getenv("NPU_U8_NHWC"));
  for (;;) {
    int32_t s; if (fread(&s, 4, 1, stdin) != 1 || s == 0) break;
    rknn_input in[NIN]; memset(in, 0, sizeof in);
    for (int k = 0; k < (int)io.n_input; k++) {
      if (fread(buf[k], 1, bsz[k], stdin) != bsz[k]) return 9;
      in[k].index = k; in[k].buf = buf[k]; in[k].size = bsz[k]; in[k].type = RKNN_TENSOR_FLOAT32;
      in[k].fmt = ia[k].n_dims == 4 ? RKNN_TENSOR_NCHW : ia[k].fmt; in[k].pass_through = 0;
      if (u8 && ia[k].n_dims == 4) {   // NPU_U8_NHWC=1: image input as uint8 NHWC (the zoo's path), from float NCHW 0-255
        int C = 3, H, W; size_t px; static unsigned char *u = NULL; H = ia[k].dims[1]; W = ia[k].dims[2]; px = (size_t)H * W;
        if (!u) u = malloc(px * C); const float *x = buf[k];
        for (size_t i = 0; i < px; i++) for (int c = 0; c < C; c++) { float v = x[c * px + i]; u[i * C + c] = v < 0 ? 0 : v > 255 ? 255 : (unsigned char)(v + 0.5f); }
        in[k].buf = u; in[k].size = px * C; in[k].type = RKNN_TENSOR_UINT8; in[k].fmt = RKNN_TENSOR_NHWC;
      }
    }
    double t0 = now_ms();
    int32_t st = rknn_inputs_set(ctx, io.n_input, in);
    if (!st) st = rknn_run(ctx, NULL);
    rknn_output o[NOUT]; memset(o, 0, sizeof o); for (int k = 0; k < (int)io.n_output; k++) o[k].want_float = 1;
    if (!st) st = rknn_outputs_get(ctx, io.n_output, o, NULL);
    float ms = (float)(now_ms() - t0); int32_t no = st ? 0 : io.n_output;
    fwrite(&st, 4, 1, stdout); fwrite(&no, 4, 1, stdout);
    for (int k = 0; k < no; k++) { int32_t c = (int32_t)(o[k].size / 4); fwrite(&c, 4, 1, stdout); fwrite(o[k].buf, 4, c, stdout); }
    fwrite(&ms, 4, 1, stdout); fflush(stdout);
    if (!st) rknn_outputs_release(ctx, io.n_output, o);
  }
  rknn_destroy(ctx); return 0;
}
