# TensorRT engines for MoDiff

This folder contains utilities to export the vanilla MoDiff UNet to ONNX and build TensorRT engines (FP32, INT8, INT4). The steps below show how to build an **ordinary FP32 TensorRT engine (no MoDiff paper optimizations/quantization)** and include it in the benchmark comparison alongside INT8/INT4 engines.

## Build a plain FP32 TensorRT engine

1. **Export ONNX** (vanilla FP32 model):
   ```bash
   python trt/export_to_onnx.py \
     --config configs/cifar10.yml \
     --output trt/export/modiff_unet_cifar10.onnx \
     --ckpt-root <path-to-checkpoints>
   ```
2. **Build FP32 TensorRT plan (no INT8 calibration, no custom scales):**
   ```bash
   python trt/build_engine.py \
     --onnx trt/export/modiff_unet_cifar10.onnx \
     --engine trt/export/modiff_unet_fp32.plan \
     --workspace-gb 8 \
     --no-int8
   ```
   This produces `trt/export/modiff_unet_fp32.plan`, representing the ordinary TensorRT FP32 graph.

## Run the benchmark with the FP32 TensorRT engine

After building the FP32 engine, include it in the comparison:
```bash
python trt/benchmark_comparison.py \
  --config configs/cifar10.yml \
  --fp32-engine trt/export/modiff_unet_fp32.plan \
  --int8-engine trt/modiff_unet_int8.plan \
  --int4-engine trt/int4_output/modiff_unet_int4.plan \
  --output-dir trt/benchmark_results
```
The script will report latency/throughput and quality metrics for:
- FP32 PyTorch baseline
- FP32 TensorRT (ordinary engine)
- INT8 TensorRT
- INT4 TensorRT
