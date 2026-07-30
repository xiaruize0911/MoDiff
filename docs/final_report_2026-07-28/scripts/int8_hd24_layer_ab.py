"""Same-process full-layer A/B for the exact T1024/hd24 Flash specialization."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/taming-transformers"))
sys.path.insert(0, os.path.dirname(__file__))

import torch

import layer_pipeline_bench as layer_bench
from int8_hd24_exact_bench import alternating_bench


def main():
    output = os.environ.get("HD24_LAYER_AB_OUT")
    os.environ["MODIFF_INT8_FLASH_HD24_EXACT"] = "0"
    model, sampler, layers = layer_bench.collect_layers("int8")
    del sampler
    matches = [
        row for row in layers
        if row["kind"] == "attention"
        and row["x_shape"] == (128, 192, 32, 32)
    ]
    if not matches:
        raise RuntimeError("T1024/C192 attention layer was not found")
    module = matches[0]["module"]
    x = torch.randn(
        *matches[0]["x_shape"], device="cuda", dtype=torch.float16
    ).contiguous(memory_format=torch.channels_last)

    def reference():
        module._int8_flash_hd24_exact = False
        return module(x)

    def candidate():
        module._int8_flash_hd24_exact = True
        return module(x)

    with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.float16):
        reference_out = reference()
        candidate_out = candidate()
        torch.cuda.synchronize()
        result = {
            "gpu": torch.cuda.get_device_name(),
            "shape": list(matches[0]["x_shape"]),
            "instances": len(matches),
            "output_bit_exact": torch.equal(reference_out, candidate_out),
            "output_max_abs_diff": (
                reference_out.float() - candidate_out.float()
            ).abs().max().item(),
            "protocol": {
                "warmups": 20,
                "rounds": 5,
                "iterations": 60,
                "same_model_same_input_alternating": True,
            },
            "layer_benchmark": alternating_bench(
                reference, candidate, 20, 5, 60),
        }
        module._int8_flash_hd24_exact = False
        ref_kernels, ref_roles, ref_gpu = layer_bench.kernel_sequence(reference)
        module._int8_flash_hd24_exact = True
        cand_kernels, cand_roles, cand_gpu = layer_bench.kernel_sequence(candidate)
        result["profile"] = {
            "reference": {
                "kernels": ref_kernels, "roles": ref_roles,
                "gpu_us_sum": ref_gpu},
            "candidate": {
                "kernels": cand_kernels, "roles": cand_roles,
                "gpu_us_sum": cand_gpu},
        }
    text = json.dumps(result, indent=2)
    if output:
        with open(output, "w") as handle:
            handle.write(text + "\n")
    print(text)
    del x, module, model


if __name__ == "__main__":
    main()
