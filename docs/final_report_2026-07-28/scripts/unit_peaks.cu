// Measure this GPU's per-unit throughputs so the "theoretical limit" for quantized attention
// is derived from what the silicon actually does under load, not from a datasheet.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define ITERS 512
#define CHK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

__global__ void k_ex2(float* out, float seed) {
  float x = seed + threadIdx.x * 1e-6f, a=0.f,b=0.f,c=0.f,d=0.f;
#pragma unroll 8
  for (int i=0;i<ITERS;++i){                       // 4 independent chains hide MUFU latency
    asm volatile("ex2.approx.f32 %0, %1;":"=f"(a):"f"(x+a*1e-8f));
    asm volatile("ex2.approx.f32 %0, %1;":"=f"(b):"f"(x+b*1e-8f));
    asm volatile("ex2.approx.f32 %0, %1;":"=f"(c):"f"(x+c*1e-8f));
    asm volatile("ex2.approx.f32 %0, %1;":"=f"(d):"f"(x+d*1e-8f));
  }
  if (threadIdx.x==2048) out[0]=a+b+c+d;           // never true; keeps work live
}
__global__ void k_ffma(float* out, float seed) {
  float a=seed,b=seed+1,c=seed+2,d=seed+3; const float m=1.0000001f;
#pragma unroll 8
  for (int i=0;i<ITERS;++i){ a=fmaf(a,m,1e-7f); b=fmaf(b,m,1e-7f); c=fmaf(c,m,1e-7f); d=fmaf(d,m,1e-7f); }
  if (threadIdx.x==2048) out[0]=a+b+c+d;
}
__global__ void k_hfma2(float* out, float seed) {
  __half2 a=__float2half2_rn(seed),b=__float2half2_rn(seed+1),c=__float2half2_rn(seed+2),d=__float2half2_rn(seed+3);
  const __half2 m=__float2half2_rn(1.0001f), n=__float2half2_rn(1e-4f);
#pragma unroll 8
  for (int i=0;i<ITERS;++i){ a=__hfma2(a,m,n); b=__hfma2(b,m,n); c=__hfma2(c,m,n); d=__hfma2(d,m,n); }
  if (threadIdx.x==2048) out[0]=__low2float(a)+__low2float(b)+__low2float(c)+__low2float(d);
}
__global__ void k_hmax2(float* out, float seed) {   // half2 max: the packed softmax primitive
  __half2 a=__float2half2_rn(seed),b=__float2half2_rn(seed+1),c=__float2half2_rn(seed+2),d=__float2half2_rn(seed+3);
  __half2 m=__float2half2_rn(0.5f);
#pragma unroll 8
  for (int i=0;i<ITERS;++i){ a=__hmax2(a,m); b=__hmax2(b,m); c=__hmax2(c,m); d=__hmax2(d,m); m=__hadd2(m,__float2half2_rn(1e-5f)); }
  if (threadIdx.x==2048) out[0]=__low2float(a)+__low2float(b)+__low2float(c)+__low2float(d);
}
__global__ void k_imma8(int* out, int seed) {
  unsigned A[4]={(unsigned)seed,0x01010101u,0x02020202u,0x03030303u}, B[2]={0x01010101u,0x02020202u};
  int C0[4]={0,0,0,0}, C1[4]={0,0,0,0};
#pragma unroll 4
  for (int i=0;i<ITERS;++i){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+r"(C0[0]),"+r"(C0[1]),"+r"(C0[2]),"+r"(C0[3]):"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+r"(C1[0]),"+r"(C1[1]),"+r"(C1[2]),"+r"(C1[3]):"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[1]),"r"(B[0]));
  }
  if (threadIdx.x==2048) out[0]=C0[0]+C1[0];
}
__global__ void k_imma4(int* out, int seed) {
  unsigned A[4]={(unsigned)seed,0x11111111u,0x22222222u,0x33333333u}, B[2]={0x11111111u,0x22222222u};
  int C0[4]={0,0,0,0}, C1[4]={0,0,0,0};
#pragma unroll 4
  for (int i=0;i<ITERS;++i){
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+r"(C0[0]),"+r"(C0[1]),"+r"(C0[2]),"+r"(C0[3]):"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]));
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+r"(C1[0]),"+r"(C1[1]),"+r"(C1[2]),"+r"(C1[3]):"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[1]),"r"(B[0]));
  }
  if (threadIdx.x==2048) out[0]=C0[0]+C1[0];
}
__global__ void k_hmma16(int* out, int seed) {      // fp16 mma, for the fp16-PV variant's peak
  unsigned A[4]={(unsigned)seed,0x3c003c00u,0x3c003c00u,0x3c003c00u}, B[2]={0x3c003c00u,0x3c003c00u};
  unsigned C0[2]={0,0}, C1[2]={0,0};
#pragma unroll 4
  for (int i=0;i<ITERS;++i){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
      :"+r"(C0[0]),"+r"(C0[1]):"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
      :"+r"(C1[0]),"+r"(C1[1]):"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[1]),"r"(B[0]));
  }
  if (threadIdx.x==2048) out[0]=(int)(C0[0]+C1[0]);
}
__global__ void k_copy(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x)
    b[i] = a[i];
}

template <class F> double timed(F f, int inner=20) {
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  for (int i=0;i<3;++i) f();
  cudaDeviceSynchronize();
  double best=1e30;
  for (int r=0;r<5;++r){
    cudaEventRecord(s); for(int i=0;i<inner;++i) f(); cudaEventRecord(e);
    cudaDeviceSynchronize(); float ms; cudaEventElapsedTime(&ms,s,e);
    double t=ms/inner/1e3; if(t<best) best=t;
  }
  cudaEventDestroy(s); cudaEventDestroy(e); return best;
}

int main(){
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0));
  int SM=p.multiProcessorCount, BL=SM*4, TH=256;
  double warps=(double)BL*TH/32, opsT=(double)ITERS*4;
  printf("GPU %s  SM=%d  clock=%.2f GHz\n", p.name, SM, p.clockRate/1e6);
  void* o; CHK(cudaMalloc(&o,64)); CHK(cudaMemset(o,0,64));

  double t;
  t=timed([&]{k_ex2<<<BL,TH>>>((float*)o,1.f);});
  double ex2=(double)BL*TH*opsT/t;
  t=timed([&]{k_ffma<<<BL,TH>>>((float*)o,1.f);});
  double f32=(double)BL*TH*opsT*2/t;
  t=timed([&]{k_hfma2<<<BL,TH>>>((float*)o,1.f);});
  double f16x2=(double)BL*TH*opsT*4/t;
  t=timed([&]{k_hmax2<<<BL,TH>>>((float*)o,1.f);});
  double hmax2=(double)BL*TH*opsT*2/t;
  t=timed([&]{k_imma8<<<BL,TH>>>((int*)o,1);});
  double i8=warps*ITERS*2*(16.0*8*32)/t;
  t=timed([&]{k_imma4<<<BL,TH>>>((int*)o,1);});
  double i4=warps*ITERS*2*(16.0*8*64)/t;
  t=timed([&]{k_hmma16<<<BL,TH>>>((int*)o,1);});
  double h16=warps*ITERS*2*(16.0*8*16)/t;

  size_t nb=1ull<<28, n16=nb/16;
  void *A,*B; CHK(cudaMalloc(&A,nb)); CHK(cudaMalloc(&B,nb)); CHK(cudaMemset(A,1,nb));
  t=timed([&]{k_copy<<<SM*8,256>>>((const uint4*)A,(uint4*)B,n16);});
  double hbm=2.0*nb/t;

  printf("\n=== 实测单元峰值 ===\n");
  printf("int8 mma  m16n8k32 : %8.1f TOPS   (MAC/s %.3e)\n", i8*2/1e12, i8);
  printf("int4 mma  m16n8k64 : %8.1f TOPS   (MAC/s %.3e)\n", i4*2/1e12, i4);
  printf("fp16 mma  m16n8k16 : %8.1f TFLOPS (MAC/s %.3e)\n", h16*2/1e12, h16);
  printf("ex2.approx.f32     : %8.3f T exp/s\n", ex2/1e12);
  printf("fp32 FFMA          : %8.1f TFLOPS\n", f32/1e12);
  printf("fp16x2 HFMA2       : %8.1f TFLOPS\n", f16x2/1e12);
  printf("fp16x2 HMAX2       : %8.1f T op/s\n", hmax2/1e12);
  printf("HBM stream (r+w)   : %8.0f GB/s\n", hbm/1e9);
  printf("\nJSON {\"int8_mac\":%.6e,\"int4_mac\":%.6e,\"fp16_mac\":%.6e,\"ex2\":%.6e,"
         "\"fp32_flop\":%.6e,\"fp16x2_flop\":%.6e,\"hmax2\":%.6e,\"hbm\":%.6e,\"sm\":%d}\n",
         i8,i4,h16,ex2,f32,f16x2,hmax2,hbm,SM);
  return 0;
}
