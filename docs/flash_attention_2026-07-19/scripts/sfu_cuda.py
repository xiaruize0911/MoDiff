import torch, time
from torch.utils.cpp_extension import load_inline
src = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>
__global__ void exp_floor_kernel(float* out, int reps){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float x = 0.001f * (idx & 1023) - 0.3f;
  float acc = 0.f;
  #pragma unroll 8
  for(int r=0;r<reps;r++){ acc += exp2f(x); x = acc*1e-9f - 0.3f; }  // reps EX2 in registers, no mem
  if(acc==-12345.f) out[idx]=acc;   // sink, never taken
}
void run_exp_floor(torch::Tensor out, int64_t threads, int64_t reps){
  int blocks = (int)(threads/256);
  exp_floor_kernel<<<blocks,256>>>(out.data_ptr<float>(), (int)reps);
}
'''
m = load_inline(name="expfloor", cpp_sources="void run_exp_floor(torch::Tensor,int64_t,int64_t);",
                cuda_sources=src, functions=["run_exp_floor"], verbose=False)
dev="cuda"
THREADS = 84*2048          # saturate the 84 SMs
REPS = 4096
out = torch.zeros(THREADS, device=dev, dtype=torch.float32)
def run(): m.run_exp_floor(out, THREADS, REPS)
for _ in range(10): run()
torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True)
s.record()
for _ in range(50): run()
e.record(); torch.cuda.synchronize()
us = s.elapsed_time(e)/50*1e3
total_exps = THREADS*REPS
rate = total_exps/(us*1e-6)
print(f"{total_exps:,} exp2 in {us:.1f} us -> {rate/1e9:.0f} G-exp2/s (register-resident, no HBM)")
N=1024*1024*1024
print(f"projected exp floor for level-0 (1.07e9 exps): {N/rate*1e6:.0f} us")
print(f"fp16 flash bar=1809us; fp16-flash MMA floor~804us")
floor=N/rate*1e6
print(f"=> softmax exp floor ~{floor:.0f}us. int8 flash ~max(402 MMA, {floor:.0f} exp)+tail. Headroom vs fp16-flash: {'good (>1.5x)' if floor<1100 else 'modest (~1.3x)' if floor<1600 else 'little'}")
