// CPU access speed of an NPU buffer (rknn_create_mem -> rknpu GEM mmap) vs plain malloc.
// gcc -O2 -o memspeed memspeed.c -I. -L. -lrknnrt -Wl,-rpath,'$ORIGIN' ; ./memspeed any.rknn
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void bench(const char*nm, unsigned char*p, size_t n){
  unsigned char*tmp=malloc(n); volatile unsigned long s=0; double t;
  memset(p,1,n);
  t=now(); for(int r=0;r<5;r++) memcpy(tmp,p,n); double rd=5*n/(now()-t)/1e6;
  t=now(); for(int r=0;r<5;r++) memcpy(p,tmp,n); double wr=5*n/(now()-t)/1e6;
  t=now(); for(int r=0;r<2;r++) for(size_t i=0;i<n;i+=64) s+=p[i]; double st=2*(n/64)/(now()-t)/1e6;
  printf("%-28s memcpy read %8.0f MB/s  write %8.0f MB/s  strided 64B loads %6.1f M/s\n",nm,rd,wr,st); free(tmp);
}
int main(int c,char**v){
  FILE*f=fopen(v[1],"rb"); fseek(f,0,2); long n=ftell(f); fseek(f,0,0); void*m=malloc(n); fread(m,1,n,f); fclose(f);
  rknn_context ctx; if(rknn_init(&ctx,m,n,0,NULL)){puts("init fail");return 1;}
  size_t sz=8<<20; unsigned char*h=malloc(sz); bench("malloc (cached)",h,sz);
  rknn_tensor_mem*tm=rknn_create_mem(ctx,sz); if(!tm){puts("create_mem fail");return 1;}
  printf("npu mem fd=%d flags=0x%x\n",tm->fd,tm->flags); bench("rknn_create_mem (NPU buffer)",(unsigned char*)tm->virt_addr,sz);
  rknn_destroy_mem(ctx,tm);
  rknn_tensor_mem*tc=rknn_create_mem2(ctx,sz,RKNN_FLAG_MEMORY_CACHEABLE); if(!tc){puts("create_mem2 fail");return 1;}
  printf("npu mem (CACHEABLE requested) fd=%d flags=0x%x\n",tc->fd,tc->flags); bench("rknn_create_mem2 CACHEABLE",(unsigned char*)tc->virt_addr,sz);
  rknn_destroy_mem(ctx,tc); rknn_destroy(ctx); return 0;
}
