#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
int main(int c, char **v) {
  FILE *f = fopen(v[1], "rb"); fseek(f, 0, 2); long n = ftell(f); fseek(f, 0, 0); void *m = malloc(n); fread(m, 1, n, f); fclose(f);
  rknn_context ctx; if (rknn_init(&ctx, m, n, 0, NULL)) return 1;
  rknn_input_output_num io; rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io);
  for (unsigned i = 0; i < io.n_input; i++) for (int q = 0; q < 2; q++) {
    rknn_tensor_attr a; memset(&a, 0, sizeof a); a.index = i;
    rknn_query(ctx, q ? RKNN_QUERY_NATIVE_INPUT_ATTR : RKNN_QUERY_INPUT_ATTR, &a, sizeof a);
    printf("%s in%u %-12s type=%s fmt=%s dims=[", q ? "native" : "user  ", i, a.name, get_type_string(a.type), get_format_string(a.fmt));
    for (unsigned d = 0; d < a.n_dims; d++) printf("%u%s", a.dims[d], d + 1 < a.n_dims ? "," : "");
    printf("] size=%u size_with_stride=%u w_stride=%u h_stride=%u pass_through=%u\n", a.size, a.size_with_stride, a.w_stride, a.h_stride, a.pass_through);
  }
  return 0;
}
