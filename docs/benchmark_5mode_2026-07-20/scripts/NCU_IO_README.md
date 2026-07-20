# Measured per-kernel, per-shape DRAM read/write IO (Nsight Compute harness)

This harness measures the **real DRAM bytes read and written by each kernel at each shape** for the
conv / linear / attention families across the quant modes — the one thing nsys memcpy cannot see
(it only sees host/device *copies*, not in-kernel DRAM traffic).

## Files
- `ncu_io_driver.py` — launches each `(family, mode, shape)` kernel **once**, wrapped in an NVTX range
  tagged `family|mode|shape` (ncu replays each kernel internally, so no warmup). Runs standalone
  (without ncu) to self-check; **validated: launches 49 tagged configs finite** (conv 15, linear 24,
  attn 10). Filter: `python ncu_io_driver.py [conv|linear|attn|all]`.
- `run_ncu_io.sh [fam]` — invokes ncu with `dram__bytes_read.sum, dram__bytes_write.sum,
  dram__bytes.sum, gpu__time_duration.sum, dram__throughput...pct` + `--nvtx --csv --page raw`, then
  parses. (Same ncu invocation shape as the repo's proven `docs/static_vs_dynamic_2026-07-16/scripts/ncu_profile.py`.)
- `parse_ncu_io.py [fam]` — maps each profiled kernel to its shape via the NVTX tag, writes
  `data/ncu_io_perkernel_<fam>.csv`: `family, mode, shape, kernel, read_MiB, write_MiB, total_MiB,
  dur_us, dram_pct_peak`.

## ⚠️ Requires unlocked GPU performance counters

On this box ncu returns **`ERR_NVGPUCTRPERM`** — counters are admin-locked
(`/proc/driver/nvidia/params: RmProfilingAdminOnly = 1`, and the container has **no `CAP_SYS_ADMIN`**;
we are uid 0 but that is not sufficient). This **cannot be changed from inside the container** — the
NVIDIA driver lives on the host. Do **one** of the following at the host / orchestration level:

**Route A — driver module param (host, persistent; recommended).** On the HOST:
```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-prof.conf'
# then reload the driver (stop all GPU processes first) OR reboot:
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
#   (or: sudo update-initramfs -u && sudo reboot)
```
Verify: `cat /proc/driver/nvidia/params | grep -i profil` → `RmProfilingAdminOnly: 0`.

**Route B — container capability (at launch time).** Recreate the container with:
```bash
docker run --cap-add SYS_ADMIN ...      # or --privileged
```
(You cannot add a capability to an already-running container — it must be recreated.)

Either route makes ncu counters accessible to this process; no code change needed.

## Usage (once unlocked)
```bash
bash docs/benchmark_5mode_2026-07-20/scripts/run_ncu_io.sh all
# -> data/ncu_io_raw_all.csv (raw ncu) and data/ncu_io_perkernel_all.csv (parsed table)
```
Produces measured read+write bytes for every kernel the model actually dispatches — e.g. the int4
conv's `scale_quantize_pack` / CUTLASS int4 conv / `scale_bias_residual_store`, the AWQ `gemm_w8a8/w4a4`,
the `flash_attn_int8/int4_vt` and `quantize_attn_qkv*` — each tagged with its shape, so you get a true
per-kernel × per-shape × {read, write} IO breakdown to replace the "unmeasurable" note in REPORT.md.

## Status
Driver validated standalone (all 49 configs launch, finite). The ncu invocation + parser follow the
repo's working `ncu_profile.py` pattern (extended for the read/write split and NVTX→shape mapping) but
**could not be exercised against live counters here** (locked). Run `run_ncu_io.sh` once counters are
unlocked; if the NVTX column name differs in your ncu version, the parser also records launch order for
fallback mapping.
