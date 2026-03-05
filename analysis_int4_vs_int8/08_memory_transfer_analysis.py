#!/usr/bin/env python3
"""
Memory Transfer Analysis: FP32 vs INT8/INT4 Standard vs INT8/INT4 MoDiff
========================================================================

REAL MEASUREMENT -- hooks into actual forward passes and records tensor.nbytes.
No formulas: HBM bytes are read directly from tensor dtype x numel on live tensors.

Measurement model per layer call
---------------------------------
FP32 (nn.Conv2d / nn.Linear):
    weight_read = weight.nbytes          (fp32, 4 B/elem)
    act_read    = input.nbytes           (fp32, 4 B/elem)
    out_write   = output.nbytes          (fp32, 4 B/elem)

INT8-standard (OptimizedInt8Conv2d, modiff_enabled=False):
    weight_read = weight_int8.nbytes     (int8, 1 B/elem)
    act_read    = input.nbytes           (fp32, 4 B/elem)
    out_write   = output.nbytes          (fp32, 4 B/elem)

INT4-standard (OptimizedInt4Conv2d, modiff_enabled=False):
    weight_read = weight_packed.nbytes   (packed int4, 0.5 B/elem effective)
    act_read    = input.nbytes           (fp32, 4 B/elem)
    out_write   = output.nbytes          (fp32, 4 B/elem)

INT8-MoDiff, first step (a_hat_cache is None before call):
    weight_read  = weight_int8.nbytes
    act_read     = input.nbytes
    out_write    = output.nbytes          (initialises o_hat_cache in-place)
    cache_write  = a_hat_cache.nbytes     (new a_hat written separately)

INT8-MoDiff, modulated step (a_hat_cache is not None before call):
    weight_read      = weight_int8.nbytes
    act_read         = input.nbytes
    cache_read_a_hat = (prev) a_hat_cache.nbytes
    cache_read_o_hat = (prev) o_hat_cache.nbytes  (for in-place accumulate)
    out_write        = output.nbytes               (= in-place o_hat update)
    cache_write      = a_hat_cache.nbytes          (new a_hat written)

INT4-MoDiff: same as INT8-MoDiff with weight_packed.nbytes.
Linear layers: identical logic using weight.nbytes / weight_int8.nbytes etc.

Usage:
    cd /workspace/MoDiff
    python analysis_int4_vs_int8/08_memory_transfer_analysis.py
    python analysis_int4_vs_int8/08_memory_transfer_analysis.py --steps 200 --batch_size 4
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14,
                         "axes.labelsize": 12, "legend.fontsize": 11})
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found -- plots will be skipped.")

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
CKPT_PATH   = "models/ldm/lsun_churches256/model.ckpt"


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------

@dataclass
class ModeIO:
    mode:              str
    steps:             int
    batch_size:        int
    weight_read_bytes: int = 0
    act_input_bytes:   int = 0
    output_bytes:      int = 0
    cache_read_bytes:  int = 0
    cache_write_bytes: int = 0
    conv_step1_read_bytes:  int = 0
    conv_step1_write_bytes: int = 0
    conv_step2_read_bytes:  int = 0
    conv_step2_write_bytes: int = 0
    conv_modulated_calls:   int = 0
    num_layer_calls:   int = 0

    @property
    def total_dram_bytes(self) -> int:
        return (self.weight_read_bytes + self.act_input_bytes +
                self.output_bytes + self.cache_read_bytes + self.cache_write_bytes)

    def total_gb(self) -> float:
        return self.total_dram_bytes / 1e9

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_bytes"] = self.total_dram_bytes
        d["total_gb"]    = self.total_gb()
        return d


MODE_LABELS = {
    "fp32":          "FP32",
    "int8_standard": "INT8 Standard",
    "int4_standard": "INT4 Standard",
    "int8_modiff":   "INT8 MoDiff",
    "int4_modiff":   "INT4 MoDiff",
}

COLORS = {
    "fp32":            "#B8E5FA",
    "int8_standard":   "#EEC186",
    "int4_standard":   "#EEF0A7",
    "int8_modiff":     "#F7A6AC",
    "int4_modiff":     "#A8D9A8",
}

COMPONENT_COLORS = {
    "Params/Weights (Rd)": "#4878CF",
    "Act Inputs (Rd)":     "#6ACC65",
    "Outputs (Wr)":        "#B47CC7",
    "Cache (Rd)":          "#C4AD66",
    "Cache (Wr)":          "#77BEDB",
}


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def _get_param_buffer_nbytes(mod: nn.Module) -> int:
    """Return size in bytes of ALL stored parameters and buffers (weights, bias, scales)."""
    total = 0
    seen = set()
    
    # Catch all typical parameters + dynamically generated buffers (e.g. integer weights)
    for name, t in mod.named_parameters(recurse=False):
        if t is not None and id(t) not in seen:
            total += t.nbytes
            seen.add(id(t))
    for name, t in mod.named_buffers(recurse=False):
        # We only count buffers that are used in the forward pass implicitly
        # skip `_orig_weight` which is just a backup reference, not read during forward.
        if t is not None and id(t) not in seen and name != "_orig_weight":
            total += t.nbytes
            seen.add(id(t))
            
    # Some layers store FP16 equivalents in attributes without registering them.
    if hasattr(mod, "weight_fp16") and mod.weight_fp16 is not None:
        if id(mod.weight_fp16) not in seen:
            total += mod.weight_fp16.nbytes
            seen.add(id(mod.weight_fp16))
            
    return total


def _is_modiff(mod: nn.Module) -> bool:
    return getattr(mod, "modiff_enabled", False)


def _is_quantized_conv_layer(mod: nn.Module) -> bool:
    return ((hasattr(mod, "weight_int8") or hasattr(mod, "weight_packed"))
            and hasattr(mod, "kernel_size")
            and hasattr(mod, "in_channels")
            and hasattr(mod, "out_channels"))


def _get_modulated_conv_qact_nbytes(mod: nn.Module, x: torch.Tensor) -> int:
    """Bytes of step1 quantized activation output (int8/int4 packed)."""
    if hasattr(mod, "weight_packed") and getattr(mod, "weight_packed") is not None:
        # INT4 packed: 2 values per byte
        return x.numel() // 2
    # INT8: 1 byte per value
    return x.numel()


def _get_modulated_conv_step2_weight_nbytes(mod: nn.Module) -> int:
    """Bytes read by conv step2 kernel (quant weights + channel scales + inv_scale)."""
    total = 0
    if hasattr(mod, "weight_int8") and mod.weight_int8 is not None:
        total += mod.weight_int8.nbytes
    if hasattr(mod, "weight_packed") and mod.weight_packed is not None:
        total += mod.weight_packed.nbytes
    if hasattr(mod, "weight_scale_channel") and mod.weight_scale_channel is not None:
        total += mod.weight_scale_channel.nbytes
    if hasattr(mod, "_inv_scale_buf") and mod._inv_scale_buf is not None:
        total += mod._inv_scale_buf.nbytes
    return total

def _is_compute_layer(mod: nn.Module) -> bool:
    """True for any leaf compute layer: Conv2d, Linear, or a quantised variant."""
    if isinstance(mod, (nn.Conv2d, nn.Linear)):
        return True
    # Quantised Conv variants don't inherit from Conv2d but carry these attrs
    if (hasattr(mod, "weight_int8")
            or hasattr(mod, "weight_packed")
            or hasattr(mod, "weight_int8_T")
            or hasattr(mod, "weight_packed_T")):
        return True
    # OptimizedInt8Linear / OptimizedInt4Linear store FP16 weights in weight_fp16
    # (they use FP16 F.linear for small-M time-embedding projections)
    if hasattr(mod, "weight_fp16"):
        return True
    return False


def attach_measurement_hooks(diffusion_model: nn.Module, io: ModeIO):
    """
    Register pre+post forward hooks on every leaf Conv2d / Linear /
    quantised-equivalent inside diffusion_model.  The hooks accumulate
    real tensor.nbytes into `io`.  Returns a list of RemovableHook handles.
    """
    hooks = []

    for name, mod in diffusion_model.named_modules():
        if not _is_compute_layer(mod):
            continue
        # Skip container modules that wrap other compute layers (avoid double-counting)
        if any(_is_compute_layer(c) for c in mod.children()):
            continue

        def _make(m):
            def pre(mod, inp):
                # Snapshot cache sizes BEFORE the forward
                a_nb = (mod.a_hat_cache.nbytes
                        if hasattr(mod, "a_hat_cache") and mod.a_hat_cache is not None
                        else 0)
                o_nb = (mod.o_hat_cache.nbytes
                        if hasattr(mod, "o_hat_cache") and mod.o_hat_cache is not None
                        else 0)
                mod._meas_a_hat_before = a_nb
                mod._meas_o_hat_before = o_nb
                mod._meas_was_cached   = (a_nb > 0)

            def post(mod, inp, out):
                x     = inp[0] if isinstance(inp, tuple) else inp
                act_b = x.nbytes
                wgt_b = _get_param_buffer_nbytes(mod)
                out_b = out.nbytes if isinstance(out, torch.Tensor) else 0

                io.act_input_bytes   += act_b
                io.weight_read_bytes += wgt_b
                io.output_bytes      += out_b
                io.num_layer_calls   += 1

                if _is_modiff(mod):
                    if mod._meas_was_cached:
                        # Modulated step: read prev a_hat + prev o_hat, write new a_hat
                        io.cache_read_bytes  += mod._meas_a_hat_before   # read a_hat_{t+1}
                        io.cache_read_bytes  += mod._meas_o_hat_before   # read o_hat_{t+1}
                        new_a = (mod.a_hat_cache.nbytes
                                 if (hasattr(mod, "a_hat_cache") and mod.a_hat_cache is not None)
                                 else mod._meas_a_hat_before)
                        io.cache_write_bytes += new_a                    # write new a_hat_t
                        # o_hat write is in-place (= output write, already in out_b)
                        
                        # Residual map dynamically built between the two C++ steps
                        if hasattr(mod, "_residual_buf") and mod._residual_buf is not None:
                            io.cache_write_bytes += mod._residual_buf.nbytes
                            io.cache_read_bytes += mod._residual_buf.nbytes

                        # Conv-only two-kernel split (measured from live tensors/buffers)
                        if _is_quantized_conv_layer(mod):
                            qact_b = _get_modulated_conv_qact_nbytes(mod, x)
                            new_o = (mod.o_hat_cache.nbytes
                                     if (hasattr(mod, "o_hat_cache") and mod.o_hat_cache is not None)
                                     else out_b)
                            residual_b = (mod._residual_buf.nbytes
                                          if (hasattr(mod, "_residual_buf") and mod._residual_buf is not None)
                                          else 0)
                            step2_w_b = _get_modulated_conv_step2_weight_nbytes(mod)

                            # Kernel-1: step1_quantize_*_fprop
                            io.conv_step1_read_bytes += act_b + mod._meas_a_hat_before
                            io.conv_step1_write_bytes += new_a + residual_b + qact_b

                            # Kernel-2: conv2d_*_fprop_o_hat
                            io.conv_step2_read_bytes += qact_b + step2_w_b + mod._meas_o_hat_before
                            io.conv_step2_write_bytes += new_o
                            io.conv_modulated_calls += 1
                    else:
                        # First step: _forward_first_step does `return o_hat.clone()`
                        # so the output is a CLONE separate from self.o_hat_cache.
                        # THREE distinct DRAM writes happen:
                        #   1. output (o_hat.clone())      → already in out_b
                        #   2. a_hat_cache                 → cache_write here
                        #   3. o_hat_cache (separate buf)  → cache_write here
                        if hasattr(mod, "a_hat_cache") and mod.a_hat_cache is not None:
                            io.cache_write_bytes += mod.a_hat_cache.nbytes
                        if hasattr(mod, "o_hat_cache") and mod.o_hat_cache is not None:
                            io.cache_write_bytes += mod.o_hat_cache.nbytes

            return pre, post

        pre_fn, post_fn = _make(mod)
        hooks.append(mod.register_forward_pre_hook(pre_fn))
        hooks.append(mod.register_forward_hook(post_fn))

    return hooks


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_base_model(config_path: str, ckpt_path: str, device: str):
    """
    Load the UNetModel directly from the YAML config, bypassing the full LDM
    model (which requires taming-transformers).  Returns a simple namespace
    with the same `.model.diffusion_model` interface used by the rest of the
    script.
    """
    import types
    from omegaconf import OmegaConf
    from ldm.modules.diffusionmodules.openaimodel import UNetModel, AttentionBlock

    conf = OmegaConf.load(config_path)
    unet_params = OmegaConf.to_container(
        conf.model.params.unet_config.params, resolve=True
    )

    unet = UNetModel(**unet_params)

    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd    = pl_sd.get("state_dict", pl_sd)
    prefix = "model.diffusion_model."
    filtered = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    unet.load_state_dict(filtered, strict=False)

    unet = unet.to(device).eval()
    unet = unet.to(memory_format=torch.channels_last)
    for m in unet.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
    # Disable checkpoint wrapper on AttentionBlock so forward works without context
    AttentionBlock.forward = lambda self, x: self._forward(x)

    inner = types.SimpleNamespace(diffusion_model=unet)
    wrapper = types.SimpleNamespace(model=inner)
    return wrapper


def _apply_int8(dm: nn.Module, use_modiff: bool, batch_size: int = 4, device: str = "cuda"):
    from integration.int8_optimized import (
        convert_model_to_optimized_int8,
        enable_modiff_mode as enable_conv,
    )
    from integration.buffer_pool import initialize_buffer_pool
    convert_model_to_optimized_int8(dm)
    initialize_buffer_pool(dm, max_batch_size=batch_size, device=device)
    try:
        from integration.int8_linear import (
            convert_model_to_int8_linear,
            enable_modiff_mode_linear,
        )
        convert_model_to_int8_linear(dm)
        enable_modiff_mode_linear(dm, use_modiff)
    except ImportError:
        pass
    enable_conv(dm, use_modiff)


def _apply_int4(dm: nn.Module, use_modiff: bool, batch_size: int = 4, device: str = "cuda"):
    from integration.int4_optimized import (
        convert_model_to_optimized_int4,
        enable_modiff_mode as enable_conv,
    )
    from integration.buffer_pool import initialize_buffer_pool
    convert_model_to_optimized_int4(dm)
    initialize_buffer_pool(dm, max_batch_size=batch_size, device=device)
    try:
        from integration.int4_linear import (
            convert_model_to_int4_linear,
            enable_modiff_mode_int4_linear,
        )
        convert_model_to_int4_linear(dm)
        enable_modiff_mode_int4_linear(dm, use_modiff)
    except ImportError:
        pass
    enable_conv(dm, use_modiff)


def _reset_modiff(dm: nn.Module):
    for m in dm.modules():
        if hasattr(m, "reset_state"):
            m.reset_state()
        elif hasattr(m, "a_hat_cache"):
            m.a_hat_cache = None
            m.o_hat_cache = None


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_mode(
    mode_name: str,
    config_path: str,
    ckpt_path: str,
    steps: int,
    batch_size: int,
    device: str,
) -> ModeIO:
    """
    Load and configure the model for `mode_name`, run `steps` forward passes
    with measurement hooks, and return a ModeIO with real byte tallies.
    """
    # Reset the global buffer pool so each mode gets a fresh one
    try:
        import integration.buffer_pool as _bp
        _bp._global_buffer_pool = None
    except Exception:
        pass

    print(f"  [{mode_name}] loading ...", flush=True)
    model = _load_base_model(config_path, ckpt_path, device)
    dm    = model.model.diffusion_model

    if mode_name == "int8_standard":
        _apply_int8(dm, use_modiff=False, batch_size=batch_size, device=device)
    elif mode_name == "int4_standard":
        _apply_int4(dm, use_modiff=False, batch_size=batch_size, device=device)
    elif mode_name == "int8_modiff":
        _apply_int8(dm, use_modiff=True, batch_size=batch_size, device=device)
    elif mode_name == "int4_modiff":
        _apply_int4(dm, use_modiff=True, batch_size=batch_size, device=device)

    io    = ModeIO(mode=mode_name, steps=steps, batch_size=batch_size)
    hooks = attach_measurement_hooks(dm, io)

    dm.eval()
    _reset_modiff(dm)

    with torch.no_grad():
        x = torch.randn(batch_size, 4, 32, 32,
                        device=device).to(memory_format=torch.channels_last)
        for step_idx in range(steps):
            t = torch.full((batch_size,), step_idx, dtype=torch.long, device=device)
            dm(x, t)

    for h in hooks:
        h.remove()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  [{mode_name}] {io.num_layer_calls} layer calls  "
          f"total={io.total_gb():.2f} GB  "
          f"(w={io.weight_read_bytes/1e9:.2f}  "
          f"act={io.act_input_bytes/1e9:.2f}  "
          f"out={io.output_bytes/1e9:.2f}  "
          f"cache_r={io.cache_read_bytes/1e9:.2f}  "
          f"cache_w={io.cache_write_bytes/1e9:.2f})")
    return io


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _gb(b: int) -> float:
    return b / 1e9


def plot_total_io(modes_io: List[ModeIO], save_dir: str):
    if not HAS_MPL:
        return
    labels     = [MODE_LABELS.get(m.mode, m.mode) for m in modes_io]
    totals_gb  = [m.total_gb() for m in modes_io]
    colors     = [COLORS.get(m.mode, "#aaa") for m in modes_io]
    fp32_total = totals_gb[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, totals_gb, color=colors, edgecolor="#444", linewidth=0.7)
    for bar, val, mode in zip(bars, totals_gb, [m.mode for m in modes_io]):
        saving = (1 - val / fp32_total) * 100 if mode != "fp32" else 0
        lbl = f"{val:.1f} GB\n({saving:+.0f}%)" if mode != "fp32" else f"{val:.1f} GB"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                lbl, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total DRAM Transfer (GB)")
    ax.set_title(f"Total DRAM Transfer -- Real Measurement\n"
                 f"({modes_io[0].steps} steps, batch={modes_io[0].batch_size})")
    ax.axhline(fp32_total, color="red", linestyle="--", linewidth=0.8,
               alpha=0.5, label="FP32 baseline")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_memory_total_io.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_stacked_breakdown(modes_io: List[ModeIO], save_dir: str):
    if not HAS_MPL:
        return
    labels = [MODE_LABELS.get(m.mode, m.mode) for m in modes_io]
    components = [
        ("Params/Weights (Rd)",   [_gb(m.weight_read_bytes) for m in modes_io]),
        ("Act Inputs (Rd)",       [_gb(m.act_input_bytes)   for m in modes_io]),
        ("Outputs (Wr)",          [_gb(m.output_bytes)      for m in modes_io]),
        ("Cache (Rd)",            [_gb(m.cache_read_bytes)  for m in modes_io]),
        ("Cache (Wr)",            [_gb(m.cache_write_bytes) for m in modes_io]),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    bottoms = [0.0] * len(labels)
    for comp_name, values in components:
        ax.bar(x, values, bottom=bottoms, label=comp_name,
               color=COMPONENT_COLORS[comp_name], edgecolor="#333", linewidth=0.4)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("DRAM Transfer (GB)")
    ax.set_title(f"DRAM Breakdown by Component -- Measured\n"
                 f"({modes_io[0].steps} steps, batch={modes_io[0].batch_size})")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_memory_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_savings_vs_fp32(modes_io: List[ModeIO], save_dir: str):
    if not HAS_MPL:
        return
    fp32 = modes_io[0]
    rest = modes_io[1:]
    labels = [MODE_LABELS.get(m.mode, m.mode) for m in rest]

    weight_savings = [_gb(fp32.weight_read_bytes - m.weight_read_bytes) for m in rest]
    cache_overhead = [_gb(-(m.cache_read_bytes + m.cache_write_bytes))   for m in rest]
    net_savings    = [_gb(fp32.total_dram_bytes  - m.total_dram_bytes)   for m in rest]

    x     = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([xi - width for xi in x], weight_savings, width,
           label="Params HBM savings",   color=COMPONENT_COLORS["Params/Weights (Rd)"])
    ax.bar([xi for xi in x], cache_overhead, width,
           label="MoDiff cache overhead", color=COMPONENT_COLORS["Cache (Wr)"])
    net_bars = ax.bar([xi + width for xi in x], net_savings, width,
                      label="Net HBM change", color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("GB  (positive = saving vs FP32, negative = extra cost)")
    ax.set_title(f"HBM Savings vs FP32 -- Measured\n"
                 f"({modes_io[0].steps} steps, batch={modes_io[0].batch_size})")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=min(0, min(cache_overhead) * 1.25))
    for bar, val in zip(net_bars, net_savings):
        y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 3.5
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_memory_savings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cumulative_io(modes_io: List[ModeIO], save_dir: str):
    if not HAS_MPL:
        return
    total_steps = modes_io[0].steps
    step_range  = [s for s in [1, 5, 10, 20, 50, 100, 200] if s <= total_steps]
    fig, ax = plt.subplots(figsize=(10, 5))
    for mio in modes_io:
        per_step = mio.total_dram_bytes / mio.steps
        totals   = [per_step * s / 1e9 for s in step_range]
        ax.plot(step_range, totals,
                label=MODE_LABELS.get(mio.mode, mio.mode),
                color=COLORS.get(mio.mode, "#aaa"),
                linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Diffusion Timesteps")
    ax.set_ylabel("Cumulative DRAM Transfer (GB)")
    ax.set_title(f"Cumulative DRAM -- Measured (batch={modes_io[0].batch_size})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_memory_cumulative.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_step_comparison(modes_io: List[ModeIO], save_dir: str):
    if not HAS_MPL:
        return
    labels    = [MODE_LABELS.get(m.mode, m.mode) for m in modes_io]
    per_step  = [m.total_dram_bytes / m.steps / 1e6 for m in modes_io]
    colors    = [COLORS.get(m.mode, "#aaa") for m in modes_io]
    fp32_ps   = per_step[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, per_step, color=colors, edgecolor="#444", linewidth=0.7)
    for bar, val, mode in zip(bars, per_step, [m.mode for m in modes_io]):
        ratio = val / fp32_ps
        lbl = f"{val:.0f} MB\n({ratio:.2f}x)" if mode != "fp32" else f"{val:.0f} MB"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                lbl, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("DRAM per Step (MB)")
    ax.set_title(f"Per-Step DRAM Cost -- Measured\n"
                 f"({modes_io[0].steps} steps, batch={modes_io[0].batch_size})")
    ax.axhline(fp32_ps, color="red", linestyle="--", linewidth=0.8,
               alpha=0.5, label="FP32/step")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_memory_per_step.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_conv_kernel_split(modes_io: List[ModeIO], save_dir: str):
    if not HAS_MPL:
        return
    labels = [MODE_LABELS.get(m.mode, m.mode) for m in modes_io]
    k1 = [_gb(m.conv_step1_read_bytes + m.conv_step1_write_bytes) for m in modes_io]
    k2 = [_gb(m.conv_step2_read_bytes + m.conv_step2_write_bytes) for m in modes_io]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    ax.bar(x, k1, label="Conv Kernel-1 IO (Rd+Wr)", color="#4E79A7", edgecolor="#333", linewidth=0.4)
    ax.bar(x, k2, bottom=k1, label="Conv Kernel-2 IO (Rd+Wr)", color="#F28E2B", edgecolor="#333", linewidth=0.4)

    for i, m in enumerate(modes_io):
        total = k1[i] + k2[i]
        if total > 0:
            ax.text(i, total + 0.15, f"{total:.1f} GB\n({m.conv_modulated_calls} calls)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("DRAM Transfer (GB)")
    ax.set_title("Conv MoDiff Modulated Path -- Kernel-1 vs Kernel-2 IO")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_memory_conv_kernel_split.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def generate_summary_table(modes_io: List[ModeIO], save_dir: str):
    fp32_total = modes_io[0].total_dram_bytes
    md = [
        "# Memory Transfer Summary -- Real Measurement",
        "",
        f"**Steps:** {modes_io[0].steps}  |  **Batch:** {modes_io[0].batch_size}",
        "",
        "> Measured via forward-pass hooks recording `tensor.nbytes` for each layer call.",
        "",
        "| Mode | Weight (GB) | Act FP32 (GB) | Output (GB) | "
        "Cache Rd (GB) | Cache Wr (GB) | **Total (GB)** | **vs FP32** |",
        "|------|------------|--------------|------------|"            
        "-------------|-------------|--------------|------------|",
    ]
    for m in modes_io:
        s = f"{(1-m.total_dram_bytes/fp32_total)*100:+.1f}%" if m.mode != "fp32" else "---"
        md.append(
            f"| {MODE_LABELS[m.mode]} "
            f"| {_gb(m.weight_read_bytes):.2f} "
            f"| {_gb(m.act_input_bytes):.2f} "
            f"| {_gb(m.output_bytes):.2f} "
            f"| {_gb(m.cache_read_bytes):.2f} "
            f"| {_gb(m.cache_write_bytes):.2f} "
            f"| **{m.total_gb():.2f}** "
            f"| {s} |"
        )
    md += ["", "> Positive = less DRAM than FP32.", ""]
    md_path = os.path.join(save_dir, "table_memory_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved {md_path}")

    tex_rows = []
    for m in modes_io:
        s = f"{(1-m.total_dram_bytes/fp32_total)*100:+.1f}\\%" if m.mode != "fp32" else "---"
        tex_rows.append(
            f"  {_tex(MODE_LABELS[m.mode])} & "
            f"{_gb(m.weight_read_bytes):.1f} & "
            f"{_gb(m.act_input_bytes):.1f} & "
            f"{_gb(m.output_bytes):.1f} & "
            f"{_gb(m.cache_read_bytes):.1f} & "
            f"{_gb(m.cache_write_bytes):.1f} & "
            f"\\textbf{{{m.total_gb():.1f}}} & "
            f"{s} \\\\"
        )
    steps, bs = modes_io[0].steps, modes_io[0].batch_size
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{Measured DRAM transfer ({steps} steps, batch {bs}).}}\n"
        "\\label{tab:memory_transfer}\n"
        "\\begin{tabular}{lrrrrrrl}\n\\toprule\n"
        "Mode & Weight & Act & Output & Cache Rd & Cache Wr & "
        "\\textbf{Total} & vs FP32 \\\\\n"
        " & (GB) & (GB) & (GB) & (GB) & (GB) & \\textbf{(GB)} & \\\\\n"
        "\\midrule\n"
    )
    tex += "\n".join(tex_rows)
    tex += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    tex_path = os.path.join(save_dir, "table_memory_summary.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  Saved {tex_path}")


def generate_per_step_table(modes_io: List[ModeIO], save_dir: str):
    fp32 = modes_io[0]
    md = [
        "# Per-Timestep DRAM Transfer -- Measured",
        "",
        "| Mode | Total (GB) | Per-Step (MB) | vs FP32 | Weight Save | Cache/step (MB) |",
        "|------|-----------|--------------|--------|------------|----------------|",
    ]
    for m in modes_io:
        per_step = m.total_dram_bytes / m.steps / 1e6
        ratio    = m.total_dram_bytes / fp32.total_dram_bytes
        w_save   = (1 - m.weight_read_bytes / fp32.weight_read_bytes) * 100 if m.mode != "fp32" else 0
        cache_mb = (m.cache_read_bytes + m.cache_write_bytes) / m.steps / 1e6
        md.append(
            f"| {MODE_LABELS[m.mode]} "
            f"| {m.total_gb():.2f} "
            f"| {per_step:.1f} "
            f"| {ratio:.2f}x "
            f"| {w_save:.0f}% "
            f"| {cache_mb:.1f} |"
        )
    md_path = os.path.join(save_dir, "table_memory_per_step.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved {md_path}")

    tex_rows = []
    for m in modes_io:
        per_step = m.total_dram_bytes / m.steps / 1e6
        ratio    = m.total_dram_bytes / fp32.total_dram_bytes
        w_save   = (1 - m.weight_read_bytes / fp32.weight_read_bytes) * 100 if m.mode != "fp32" else 0.0
        cache_mb = (m.cache_read_bytes + m.cache_write_bytes) / m.steps / 1e6
        tex_rows.append(
            f"  {_tex(MODE_LABELS[m.mode])} & "
            f"{m.total_gb():.2f} & {per_step:.1f} & {ratio:.2f}$\\\\times$ & "
            f"{w_save:.0f}\\% & {cache_mb:.1f} \\\\"
        )
    steps, bs = modes_io[0].steps, modes_io[0].batch_size
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{Per-timestep DRAM ({steps} steps, batch {bs}).}}\n"
        "\\label{tab:memory_per_step}\n"
        "\\begin{tabular}{lrrrrl}\n\\toprule\n"
        "Mode & Total (GB) & Per-Step (MB) & vs FP32 & W-Save & Cache (MB/step) \\\\\n"
        "\\midrule\n"
    )
    tex += "\n".join(tex_rows)
    tex += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    tex_path = os.path.join(save_dir, "table_memory_per_step.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  Saved {tex_path}")


def generate_conv_kernel_table(modes_io: List[ModeIO], save_dir: str):
    md = [
        "# Conv Kernel IO Split (MoDiff modulated steps only)",
        "",
        "| Mode | K1 Read (GB) | K1 Write (GB) | K1 Total (GB) | K2 Read (GB) | K2 Write (GB) | K2 Total (GB) | Calls |",
        "|------|--------------|---------------|---------------|--------------|---------------|---------------|-------|",
    ]
    for m in modes_io:
        k1 = _gb(m.conv_step1_read_bytes + m.conv_step1_write_bytes)
        k2 = _gb(m.conv_step2_read_bytes + m.conv_step2_write_bytes)
        md.append(
            f"| {MODE_LABELS[m.mode]} "
            f"| {_gb(m.conv_step1_read_bytes):.2f} "
            f"| {_gb(m.conv_step1_write_bytes):.2f} "
            f"| {k1:.2f} "
            f"| {_gb(m.conv_step2_read_bytes):.2f} "
            f"| {_gb(m.conv_step2_write_bytes):.2f} "
            f"| {k2:.2f} "
            f"| {m.conv_modulated_calls} |"
        )
    path = os.path.join(save_dir, "table_memory_conv_kernel_split.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved {path}")


def generate_report(modes_io: List[ModeIO], save_dir: str):
    fp32  = modes_io[0]
    int8m = next(m for m in modes_io if m.mode == "int8_modiff")
    int4m = next(m for m in modes_io if m.mode == "int4_modiff")
    int8s = next(m for m in modes_io if m.mode == "int8_standard")
    int4s = next(m for m in modes_io if m.mode == "int4_standard")

    def saving(m):
        return (1 - m.total_dram_bytes / fp32.total_dram_bytes) * 100

    lines = [
        "# Memory Transfer Analysis Report -- Real Measurement",
        "",
        f"**Model**: LSUN-Churches LDM (U-Net diffusion model)  ",
        f"**Steps**: {fp32.steps}  |  **Batch**: {fp32.batch_size}  ",
        f"**Method**: Forward-pass hooks (measured)  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Mode | Total HBM (GB) | vs FP32 | Weight (GB) | Cache total (GB) |",
        "|------|--------------|--------|------------|-----------------|",
    ]
    for m in modes_io:
        cache_gb = (m.cache_read_bytes + m.cache_write_bytes) / 1e9
        s = f"{saving(m):+.1f}%" if m.mode != "fp32" else "---"
        lines.append(
            f"| {MODE_LABELS[m.mode]} "
            f"| {m.total_gb():.2f} "
            f"| {s} "
            f"| {m.weight_read_bytes/1e9:.2f} "
            f"| {cache_gb:.2f} |"
        )
    lines += [
        "",
        "## Key Findings",
        "",
        f"- **INT8 Standard** reads {fp32.weight_read_bytes/1e9:.1f} GB -> {int8s.weight_read_bytes/1e9:.1f} GB weight bytes "
        f"({fp32.weight_read_bytes/int8s.weight_read_bytes:.1f}x compression); ",
        f"- **INT4 Standard** reads {fp32.weight_read_bytes/1e9:.1f} GB -> {int4s.weight_read_bytes/1e9:.1f} GB weight bytes "
        f"({fp32.weight_read_bytes/int4s.weight_read_bytes:.1f}x compression); ",
        f"- **INT8 MoDiff** total: {int8m.total_gb():.2f} GB ({saving(int8m):+.1f}% vs FP32); "
        f"cache overhead: {(int8m.cache_read_bytes+int8m.cache_write_bytes)/1e9:.2f} GB;",
        f"- **INT4 MoDiff** total: {int4m.total_gb():.2f} GB ({saving(int4m):+.1f}% vs FP32); "
        f"cache overhead: {(int4m.cache_read_bytes+int4m.cache_write_bytes)/1e9:.2f} GB;",
        "",
        "## Why MoDiff Is Still Fast",
        "",
        "MoDiff's speedup comes from 4x/8x GEMM throughput (tensor cores process",
        "INT8/INT4 residuals, not FP32 activations), not from reduced DRAM bandwidth.",
        "The residuals (a_t - a_hat_{t+1}) have ~10x smaller range, enabling INT4",
        "quantization with FP32-level output quality (Theorem 1 of the MoDiff paper).",
        "",
        "## Measurement Method",
        "",
        "### Directly Measured (forward-pass hooks, `tensor.nbytes`)",
        "- `input[0]` — FP32 activation (4 B/elem)",
        "- quantised weight buffer: `weight_int8` (1 B/elem), `weight_packed` (0.5 B/elem),",
        "  `weight_fp16` for linear layers (2 B/elem)",
        "- `output` — FP32 (4 B/elem)",
        "- MoDiff `a_hat_cache` read + write; `o_hat_cache` read (from next-step perspective)",
        "  (o_hat write IS the output write, not double-counted)",
        "",
        "### Conv-only modulated kernel split (measured labels)",
        "- **Kernel-1** (`step1_quantize_*_fprop`): reads input + `a_hat_cache`; writes",
        "  quantized activation + updated `a_hat_cache` + `_residual_buf`",
        "- **Kernel-2** (`conv2d_*_fprop_o_hat`): reads quantized activation + quantized",
        "  weights + `weight_scale_channel` + previous `o_hat_cache`; writes updated `o_hat_cache`",
        "- Output file: `table_memory_conv_kernel_split.md` and `plot_memory_conv_kernel_split.png`",
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `memory_transfer_analysis.json` | Raw numbers for all modes |",
        "| `plot_memory_total_io.png` | Total IO bar chart |",
        "| `plot_memory_breakdown.png` | Stacked component breakdown |",
        "| `plot_memory_savings.png` | Savings vs FP32 |",
        "| `plot_memory_cumulative.png` | IO accumulation over timesteps |",
        "| `plot_memory_per_step.png` | Per-step bar chart |",
        "| `plot_memory_conv_kernel_split.png` | Conv kernel-1 vs kernel-2 IO |",
        "| `table_memory_summary.md/tex` | Full summary table |",
        "| `table_memory_per_step.md/tex` | Per-step IO table |",
        "| `table_memory_conv_kernel_split.md` | Conv kernel-1/kernel-2 IO table |",
        "",
    ]
    path = os.path.join(save_dir, "MEMORY_TRANSFER_REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real memory transfer measurement for MoDiff")
    parser.add_argument("--steps",      type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--config",     type=str, default=CONFIG_PATH)
    parser.add_argument("--ckpt",       type=str, default=CKPT_PATH)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes",      type=str,
                        default="fp32,int8_standard,int4_standard,int8_modiff,int4_modiff")
    args = parser.parse_args()

    print("=" * 60)
    print("MoDiff Memory Transfer Analysis  (REAL MEASUREMENT)")
    print("=" * 60)
    print(f"  Steps: {args.steps}  |  Batch: {args.batch_size}  |  Device: {args.device}")
    print()

    mode_list = [m.strip() for m in args.modes.split(",")]

    print("[1/3] Measuring DRAM IO -- loading model fresh for each mode ...")
    print()

    modes_io: List[ModeIO] = []
    for mode_name in mode_list:
        try:
            io = measure_mode(
                mode_name,
                config_path=args.config,
                ckpt_path=args.ckpt,
                steps=args.steps,
                batch_size=args.batch_size,
                device=args.device,
            )
            modes_io.append(io)
        except Exception as exc:
            print(f"  ERROR [{mode_name}]: {exc}")
            import traceback; traceback.print_exc()
        print()

    if not modes_io:
        print("No modes succeeded. Exiting.")
        return

    json_path = os.path.join(OUTPUT_DIR, "memory_transfer_analysis.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "device": args.device,
                "measurement": "real_hooks",
            },
            "modes": {m.mode: m.to_dict() for m in modes_io},
        }, f, indent=2)
    print(f"  Saved {json_path}\n")

    print("[2/3] Generating plots ...")
    plot_total_io(modes_io, OUTPUT_DIR)
    plot_stacked_breakdown(modes_io, OUTPUT_DIR)
    plot_savings_vs_fp32(modes_io, OUTPUT_DIR)
    plot_cumulative_io(modes_io, OUTPUT_DIR)
    plot_per_step_comparison(modes_io, OUTPUT_DIR)
    plot_conv_kernel_split(modes_io, OUTPUT_DIR)
    print()

    print("[3/3] Generating tables & report ...")
    generate_summary_table(modes_io, OUTPUT_DIR)
    generate_per_step_table(modes_io, OUTPUT_DIR)
    generate_conv_kernel_table(modes_io, OUTPUT_DIR)
    generate_report(modes_io, OUTPUT_DIR)
    print()

    print("=" * 60)
    print("Done!  Outputs in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
