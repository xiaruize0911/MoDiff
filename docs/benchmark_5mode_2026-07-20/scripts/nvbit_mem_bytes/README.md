# NVBit `mem_bytes` tool — measured per-kernel DRAM read/write bytes (no perf counters)

Custom NVBit tool that counts GLOBAL-memory read/write bytes per kernel by instrumenting SASS at
runtime (via the CUDA driver callback API) — **no `CAP_SYS_ADMIN` / no hardware perf counters**, so it
works on this counter-locked box where `ncu`/CUPTI-metrics/DCGM/`nsys --gpu-metrics` all fail with
`ERR_NVGPUCTRPERM`. Because it's SASS-level, it covers **every** kernel including CUTLASS conv and cuDNN.

- `inject_funcs.cu` — device: per global ld/st, `atomicAdd(active_threads * access_size)` to a read or
  write counter (one atomic per warp; classified by opcode LD=read, ST/RED/ATOM=write).
- `mem_bytes.cu` — host tool: instruments GLOBAL loads/stores, resets counters per launch, prints
  `MEMBYTES read=<B> write=<B> blocks=<n> kernel=<name>` for each kernel launched inside a
  `cuProfilerStart/Stop` region (`ACTIVE_FROM_START=0`), plus a `MEMBYTES_TOTAL` at exit.

**Validation:** an fp16 in-place `add_` on an 8192² tensor measured `read=134217728 write=134217728` =
exactly 8192²×2 bytes (byte-perfect).

## Build (needs an NVBit release)
```bash
# download the release (v1.8 used here) and drop this tool into its tools/ dir:
curl -sL -o nvbit.tar.bz2 https://github.com/NVlabs/NVBit/releases/download/v1.8/nvbit-Linux-x86_64-1.8.tar.bz2
tar xjf nvbit.tar.bz2                       # -> nvbit_release_x86_64/
cp -r <this dir> nvbit_release_x86_64/tools/mem_bytes
cd nvbit_release_x86_64/tools/mem_bytes && source /workspace/MoDiff/setup_cuda_env.sh && make   # -> mem_bytes.so
```

## Run (from /workspace/MoDiff)
`../run_nvbit_io.sh` drives `../nvbit_io_driver.py` one config per process under
`ACTIVE_FROM_START=0 LD_PRELOAD=mem_bytes.so` and parses to
`data/nvbit_io_total.csv` (per-config read/write) + `data/nvbit_io_perkernel.csv` (per-kernel).
Point `TOOL=` in `run_nvbit_io.sh` at your built `mem_bytes.so`.
