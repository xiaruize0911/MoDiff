"""
MoDiff Error-Compensated Modulation for AttentionBlock — CUTLASS INT8 GEMM edition.

Applies the temporal delta-quantization trick from the MoDiff paper
(Gao et al., ICML 2025) to the two linear sub-layers of AttentionBlock:
    - self.qkv    : Conv1d(C, 3C, 1)  — linear in norm(x)
    - self.proj_out : Conv1d(C, C, 1)  — linear in attention output h

MoDiff equations (applied to each Conv1d layer):
    t=T (first step, reset):
        a_hat_T = Q(input_T)                        -- quantize input
        o_hat_T = INT8_GEMM(a_hat_T) + bias         -- cache output
    t<T (modulated steps):
        delta_t = Q(input_t - a_hat_{t+1})           -- quantize residual
        o_hat_t = INT8_GEMM(delta_t) + o_hat_{t+1}   -- Eq. ec6
        a_hat_t ≈ input_t                            -- update cache

Primary backend: CUTLASS (same fused kernel as ResBlock INT8)
    Conv1d(C_in, C_out, ks=1) is reshaped to Conv2d(C_in, C_out, ks=1×1)
    and delegated to OptimizedInt8Conv2d.  The CUTLASS kernel fuses
    subtract+absmax+quantize (step1) and INT8_GEMM+dequantize+accumulate
    (step2) into just 2 CUDA launches — identical to ResBlock INT8 timing.

Fallback backend: torch._int_mm (cuBLAS INT8 GEMM) without kernel fusion.
    ~12 separate kernel launches due to broken torch.compile in this env.
    Slower than FP16 F.conv1d; kept only for environments without CUTLASS.
"""

import torch
import torch.nn as nn
from typing import Optional

# ---------------------------------------------------------------------------
# CUTLASS availability check (delegates to OptimizedInt8Conv2d)
# ---------------------------------------------------------------------------
try:
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d as _OInt8, HAS_CUTLASS as _HC
    HAS_CUTLASS_ATTN = _HC
except Exception:
    HAS_CUTLASS_ATTN = False
    _OInt8 = None

# ---------------------------------------------------------------------------
# Fused layout-transpose / dtype-cast kernels (K1+K2 and K7+K8 fusion)
# + Fused pre-GEMM kernel (K1+K2+K3 fusion)
# ---------------------------------------------------------------------------
try:
    import modiff_cutlass as _mc
    _fp16_ncw_to_fp32_cl    = _mc.fp16_ncw_to_fp32_cl
    _fp32_cl_to_fp16_ncw    = _mc.fp32_cl_to_fp16_ncw
    _HAS_FUSED_CAST         = True
    # Fused K1+K2+K3: FP16 NCW + a_hat → INT8 CL (single kernel)
    _fp16_ncw_delta_to_int8_cl = getattr(_mc, 'fp16_ncw_delta_to_int8_cl', None)
    _HAS_FUSED_PREQUANT     = _fp16_ncw_delta_to_int8_cl is not None
except (ImportError, AttributeError):
    _HAS_FUSED_CAST         = False
    _HAS_FUSED_PREQUANT     = False
    _fp16_ncw_to_fp32_cl        = None
    _fp32_cl_to_fp16_ncw        = None
    _fp16_ncw_delta_to_int8_cl  = None

# ---------------------------------------------------------------------------
# Check INT8 GEMM availability (fallback when CUTLASS unavailable)
# ---------------------------------------------------------------------------
try:
    _probe = torch.zeros(32, 32, dtype=torch.int8, device='cuda')
    torch._int_mm(_probe, _probe)
    _HAS_INT_MM = True
    del _probe
except (AttributeError, RuntimeError):
    _HAS_INT_MM = False


# ---------------------------------------------------------------------------
# torch.compile'd hot-path functions
# Compiling as module-level functions (not methods) avoids recompilation per
# instance, and allows Triton to fuse all surrounding elementwise ops into
# 2–3 kernels instead of the default ~12 separate launches.
# ---------------------------------------------------------------------------

def _int8_gemm_impl(
    x_int8: torch.Tensor,          # [B, C_in, L]  int8
    x_scale: torch.Tensor,         # scalar
    w_int8_t: torch.Tensor,        # [C_in, C_out] int8  (pre-transposed)
    w_scale: torch.Tensor,         # scalar
    bias: Optional[torch.Tensor],  # [C_out] or None
) -> torch.Tensor:                 # [B, C_out, L]  float32
    B, C_in, L = x_int8.shape
    x_2d = x_int8.permute(0, 2, 1).reshape(B * L, C_in).contiguous()
    out_i32 = torch._int_mm(x_2d, w_int8_t)                   # [B*L, C_out] int32
    out_f32 = out_i32.to(torch.float32) * (x_scale * w_scale)  # dequantize
    if bias is not None:
        out_f32 = out_f32 + bias                               # [C_out] broadcast
    return out_f32.reshape(B, L, -1).permute(0, 2, 1).contiguous()


def _first_step_impl(
    x_f16: torch.Tensor,
    w_int8_t: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:   # returns [B, C_out, L] float32
    x_f32  = x_f16.float()
    scale  = x_f32.abs().max().clamp(1e-8) / 127.0
    x_int8 = (x_f32 / scale).round().clamp(-127.0, 127.0).to(torch.int8)
    return _int8_gemm_impl(x_int8, scale, w_int8_t, w_scale, bias)


def _modulated_step_impl(
    x_f16:    torch.Tensor,   # [B, C_in, L]  current input (FP16)
    a_hat:    torch.Tensor,   # [B, C_in, L]  cached input from prev step (FP16)
    o_hat:    torch.Tensor,   # [B, C_out, L] cached output from prev step (FP16)
    w_int8_t: torch.Tensor,   # [C_in, C_out] int8
    w_scale:  torch.Tensor,   # scalar
    bias: Optional[torch.Tensor],
) -> torch.Tensor:            # [B, C_out, L] FP16
    delta  = (x_f16 - a_hat).float()          # small residual, high temporal coherence
    scale  = delta.abs().max().clamp(1e-8) / 127.0
    d_int8 = (delta / scale).round().clamp(-127.0, 127.0).to(torch.int8)
    out_f32 = _int8_gemm_impl(d_int8, scale, w_int8_t, w_scale, bias)
    return out_f32.half() + o_hat             # Eq. ec6: A(Q(delta)) + o_hat_{t+1}


# Attempt to compile both hot paths via torch.compile (Triton backend).
# Some environments have broken inductor installations; we fall back gracefully.
_compiled_first_step     = _first_step_impl
_compiled_modulated_step = _modulated_step_impl
if _HAS_INT_MM:
    try:
        _compiled_first_step     = torch.compile(_first_step_impl,     dynamic=True, fullgraph=False)
        _compiled_modulated_step = torch.compile(_modulated_step_impl, dynamic=True, fullgraph=False)
        _USE_COMPILE = True
    except Exception as _compile_err:
        import warnings as _w
        _w.warn(
            f"torch.compile unavailable ({_compile_err.__class__.__name__}: {_compile_err}). "
            "MoDiffConv1d will use uncompiled torch._int_mm (~12 kernel launches / call). "
            "Install a clean PyTorch build to enable Triton fusion."
        )
        _USE_COMPILE = False
else:
    _USE_COMPILE = False


# ---------------------------------------------------------------------------
# MoDiffConv1dCUTLASS — primary fast path (CUTLASS fused kernel)
# ---------------------------------------------------------------------------

_GRAPH_WARMUP_STEPS = 3   # eager steps before CUDA graph capture


class MoDiffConv1dCUTLASS(nn.Module):
    """
    Fast drop-in for Conv1d(ks=1) using the proven CUTLASS ResBlock kernel.

    Conv1d(C_in, C_out, ks=1)  ==  Conv2d(C_in, C_out, ks=1×1).
    We reshape [B, C, L] → [B*L, C, 1, 1], run OptimizedInt8Conv2d, then
    reshape back to [B, C_out, L].

    Kernel count for modulated step (hot path, eager):
        1. permute+contiguous          : [B,C,L] FP16 → [B*L,C,1,1] FP16
        2. float()                     : FP16 → FP32
        3. channels_last copy          : fix strides for CUTLASS (H=W=1 false positive)
        4. step1_quantize_fprop        : sub_absmax_scale + quantize_update_ahat  [2 CUDA kernels]
        5. conv2d_int8_fprop_o_hat     : INT8 GEMM + dequant + accumulate
        6. output permute+contiguous   : [B*L,C_out,1,1] FP32 → [B,C_out,L] FP32
        7. to(fp16)                    : FP32 → FP16
        Total: 8 CUDA kernels / 7 Python dispatches  (eager)

    With CUDA graph (Option B):
        After _GRAPH_WARMUP_STEPS eager calls, the 8 CUDA kernels are captured
        into a CUDA graph.  Each subsequent modulated step costs only:
        1. copy_()   : feed new input into static buffer
        2. g.replay(): replay all 8 captured kernels without Python overhead
        Python dispatches: 2 vs 7  (saves ~5 dispatches × ~1.5µs × 42 calls ≈ 0.3ms/step)
    """

    def __init__(self, conv1d: nn.Conv1d, layer_name: str = ""):
        super().__init__()
        self._layer_name = layer_name

        from integration.kernels.int8_optimized import OptimizedInt8Conv2d

        # Convert Conv1d(C_in, C_out, ks=1) → equivalent Conv2d(ks=1×1)
        conv2d = nn.Conv2d(
            conv1d.in_channels, conv1d.out_channels,
            kernel_size=1, bias=(conv1d.bias is not None),
        )
        conv2d.weight.data = conv1d.weight.data.reshape(
            conv1d.out_channels, conv1d.in_channels, 1, 1
        ).clone()
        if conv1d.bias is not None:
            conv2d.bias.data = conv1d.bias.data.clone()

        self.int8conv2d = OptimizedInt8Conv2d(conv2d, layer_name=layer_name)
        self.int8conv2d.modiff_enabled = True   # arm MoDiff immediately

        # Minimal stats (delta_ratio diagnostics, matches MoDiffConv1d interface)
        self._stat_calls: int = 0

        # CUDA graph state (Option B)
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_x_static: Optional[torch.Tensor] = None    # static input buffer
        self._graph_out_ref:   Optional[torch.Tensor] = None    # output tensor inside graph pool
        self._graph_shape: Optional[tuple] = None               # (B, C, L) captured for
        self._graph_warmup_n: int = 0
        self._graph_capture_failed: bool = False

    # ------------------------------------------------------------------
    # Eager forward (always correct; used during first step + warmup)
    # ------------------------------------------------------------------
    def _forward_eager(self, x: torch.Tensor, B: int, C: int, L: int) -> torch.Tensor:
        if _HAS_FUSED_CAST and x.dtype == torch.float16:
            int8c = self.int8conv2d

            # ---- Fused K1+K2+K3 hot path (calibrated modulated step only) ----
            # Replaces:  _fp16_ncw_to_fp32_cl  (1 kernel)
            #          + step1_static_quantize_fprop  (1 kernel)
            # With a single tiled kernel → saves 1 kernel / call × 42 calls/step
            #
            # fp16_ncw_delta_to_int8_cl's CUDA kernel only has an FP32 a_hat_cache
            # code path (see csrc/kernels/layout_transform.cu), but a *calibrated*
            # OptimizedInt8Conv2d switches its cache to FP16 once calibration
            # finishes (see cache_dtype in int8_optimized.py). So this fused
            # shortcut is only valid pre-calibration-switch / when the cache
            # happens to still be FP32; otherwise fall through to the generic
            # path below, which dispatches on cache dtype correctly.
            if (_HAS_FUSED_PREQUANT
                    and not int8c.is_first_step
                    and int8c.is_calibrated
                    and int8c.a_hat_cache is not None
                    and int8c.a_hat_cache.dtype == torch.float32
                    and int8c._cached_alpha_tensor is not None):
                import modiff_cutlass as _mc_
                # K1+K2+K3 fused: FP16 NCW → INT8 CL + update a_hat_cache (1 kernel)
                x_int8 = _fp16_ncw_delta_to_int8_cl(
                    x, int8c.a_hat_cache, int8c.static_input_scale, B, C, L
                )
                # K4: CUTLASS INT8 GEMM + dequant + accumulate into o_hat_cache (1 kernel)
                _mc_.conv2d_int8_fprop_o_hat(
                    x_int8, int8c.weight_int8,
                    int8c._cached_alpha_tensor.view(1),
                    int8c.weight_scale_channel.view(-1),
                    int8c.o_hat_cache,
                    1, 1, 0, 0, 1, 1,
                )
                # K7+K8 fused: FP32 CL → FP16 NCW (1 kernel)
                C_out = int8c.o_hat_cache.shape[1]
                return _fp32_cl_to_fp16_ncw(int8c.o_hat_cache, B, C_out, L)

            # ---- Original 4-kernel path (first step or not yet calibrated) ----
            # K1+K2 fused: FP16 NCW → FP32 CL (1 kernel)
            x_cl = _fp16_ncw_to_fp32_cl(x, B, C, L)      # [B*L, C, 1, 1] FP32 CL
            out_4d = int8c(x_cl)                           # [B*L, C_out, 1, 1] FP32
            C_out = out_4d.shape[1]
            return _fp32_cl_to_fp16_ncw(out_4d, B, C_out, L)  # [B, C_out, L] FP16
        else:
            # Fallback (non-FP16 input, e.g. FP32 warmup pass, or no fused kernels):
            # K1+K2: permute FP16 [B,C,L]→[B,L,C] + cast to FP32 in one contiguous pass.
            # Doing float() here means OptimizedInt8Conv2d.forward skips its own K2 cast.
            x_fp32 = x.permute(0, 2, 1).contiguous().float().view(B * L, C, 1, 1)
            # Skip K3: for H=W=1, NCHW strides (C,1,1,1) and channels-last strides (C,1,C,C)
            # produce identical physical memory layout.  Use as_strided to set channels-last
            # strides without any copy, so OptimizedInt8Conv2d skips its copy.
            x_cl = x_fp32.as_strided(x_fp32.shape, (C, 1, C, C))
            out_4d = self.int8conv2d(x_cl)          # [B*L, C_out, 1, 1]
            C_out = out_4d.shape[1]
            return out_4d.view(B, L, C_out).permute(0, 2, 1).contiguous().to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L]  →  [B, C_out, L]"""
        B, C, L = x.shape

        # First step (warm-up / cache initialisation): always run eagerly
        if self.int8conv2d.is_first_step:
            return self._forward_eager(x, B, C, L)

        # --- CUDA graph hot path ---
        if (self._cuda_graph is not None
                and (B, C, L) == self._graph_shape
                and x.dtype == self._graph_x_static.dtype):
            # Feed new input into the static buffer, then replay captured kernels.
            # The dtype check prevents a cross-dtype copy_ that would add an extra
            # GPU kernel (negating the savings) when warmup runs in FP32 but
            # timed inference runs in FP16 autocast.
            self._graph_x_static.copy_(x)
            self._cuda_graph.replay()
            return self._graph_out_ref

        # --- Eager path (also does graph-capture bookkeeping) ---
        out = self._forward_eager(x, B, C, L)

        if not self._graph_capture_failed and self._cuda_graph is None:
            self._graph_warmup_n += 1
            if self._graph_warmup_n >= _GRAPH_WARMUP_STEPS:
                self._try_capture_graph(x, B, C, L)

        return out

    # ------------------------------------------------------------------
    # CUDA graph capture (called once, after warmup)
    # ------------------------------------------------------------------
    def _try_capture_graph(self, x_sample: torch.Tensor,
                           B: int, C: int, L: int) -> None:
        try:
            import warnings
            # Always capture with float16: model inference runs with fp16 autocast
            # but the benchmark warmup pass has no autocast (fp32 activations).
            # Capturing in fp16 ensures the timed-pass copy_ is a same-dtype memcpy
            # rather than a cross-dtype cast that would add an extra GPU kernel and
            # erase the graph's Python-dispatch savings.
            x_fp16 = x_sample.to(torch.float16)
            self._graph_x_static = x_fp16.clone()
            g = torch.cuda.CUDAGraph()

            # Use a side stream so capture doesn't interfere with the default stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())

            # Extra warm-up on the capture stream (required by CUDA graph API)
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._forward_eager(self._graph_x_static, B, C, L)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            # Capture
            with torch.cuda.stream(s):
                with torch.cuda.graph(g):
                    self._graph_out_ref = self._forward_eager(
                        self._graph_x_static, B, C, L)

            self._cuda_graph   = g
            self._graph_shape  = (B, C, L)

        except Exception as exc:
            import warnings
            warnings.warn(
                f"[MoDiffAttn] CUDA graph capture failed for layer "
                f"'{self._layer_name}' (shape {(B,C,L)}): "
                f"{exc.__class__.__name__}: {exc}. "
                "Falling back to eager mode."
            )
            self._cuda_graph          = None
            self._graph_x_static      = None
            self._graph_out_ref       = None
            self._graph_capture_failed = True

    def reset_cache(self):
        """Reset MoDiff caches between DDIM samples.

        Must destroy any captured CUDA graph here: OptimizedInt8Conv2d's
        first-step path (run on the very next forward call, not by this
        reset itself) does `self.a_hat_cache = a_hat.to(...).contiguous(...)`
        and the same for o_hat_cache -- a *reassignment* to new tensor
        objects, not an in-place write. A previously captured graph has the
        old tensors' addresses baked in, so once those old tensors are
        reassigned away (and their memory is freed/reused by the caching
        allocator) a stale graph replay reads/writes freed memory --
        surfacing as an illegal memory access on the first replay after a
        reset. Clearing graph state forces a fresh capture (still gated by
        _GRAPH_WARMUP_STEPS eager calls) against the post-reset tensors.
        """
        self.int8conv2d.reset_state()
        self._cuda_graph = None
        self._graph_x_static = None
        self._graph_out_ref = None
        self._graph_shape = None
        self._graph_warmup_n = 0
        self._graph_capture_failed = False

    @property
    def layer_name(self) -> str:
        return self._layer_name

    @property
    def modiff_enabled(self) -> bool:
        return self.int8conv2d.modiff_enabled

    @modiff_enabled.setter
    def modiff_enabled(self, val: bool):
        self.int8conv2d.modiff_enabled = val


# ---------------------------------------------------------------------------
# MoDiff Conv1d with torch._int_mm — fallback when CUTLASS unavailable
# ---------------------------------------------------------------------------

class MoDiffConv1d(nn.Module):
    """
    Drop-in replacement for Conv1d(C_in, C_out, kernel_size=1) with:
      - Pre-quantized INT8 weights (init time only)
      - cuBLAS INT8 GEMM via torch._int_mm
      - torch.compile fused quantize/dequantize (2–3 kernels vs ~12 uncompiled)
      - MoDiff temporal delta caching (Eq. ec5/ec6, Gao et al. ICML 2025)
    """

    def __init__(self, conv1d: nn.Conv1d, act_bits: int = 8, layer_name: str = ""):
        super().__init__()
        self.act_bits    = act_bits
        self.layer_name  = layer_name
        self.use_int8_gemm = _HAS_INT_MM and act_bits == 8
        self.C_out = conv1d.weight.shape[0]
        self.C_in  = conv1d.weight.shape[1]

        # ------------------------------------------------------------------
        # Pre-quantize weights to INT8 at init (one-time cost, not hot path)
        # ------------------------------------------------------------------
        w_fp32  = conv1d.weight.data.float().view(self.C_out, self.C_in)
        w_amax  = w_fp32.abs().max().clamp(min=1e-8)
        w_scale = w_amax / 127.0
        w_int8  = (w_fp32 / w_scale).round_().clamp_(-127, 127).to(torch.int8)

        # Pre-transpose so _int_mm can receive [C_in, C_out] without a runtime transpose
        self.register_buffer("w_int8_t", w_int8.t().contiguous())   # [C_in, C_out] int8
        self.register_buffer("w_scale",  w_scale.view(1))            # scalar CUDA tensor

        # Fallback FP16 weights for environments without torch._int_mm
        if not self.use_int8_gemm:
            self.register_buffer("weight_fp16", conv1d.weight.data.half())

        if conv1d.bias is not None:
            self.register_buffer("bias_fp32", conv1d.bias.data.float())  # [C_out]
        else:
            self.bias_fp32 = None

        # ------------------------------------------------------------------
        # MoDiff state (reset between DDIM samples via reset_cache())
        # ------------------------------------------------------------------
        self.modiff_enabled: bool = True
        self.is_first_step:  bool = True
        self.a_hat_cache: Optional[torch.Tensor] = None   # [B, C_in, L]  FP16
        self.o_hat_cache: Optional[torch.Tensor] = None   # [B, C_out, L] FP16

        # Stats: populated only for first 200 modulated steps (gated to avoid hot-path cost)
        self._stat_calls:     int   = 0
        self._stat_full_rms:  float = 0.0
        self._stat_delta_rms: float = 0.0

    # ------------------------------------------------------------------
    # FP16 fallback (used when torch._int_mm is unavailable)
    # ------------------------------------------------------------------

    def _fp16_conv(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        out = F.conv1d(x.half(), self.weight_fp16).float()
        if self.bias_fp32 is not None:
            out = out + self.bias_fp32.view(1, -1, 1)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, L]  →  [B, C_out, L]"""
        x_f16 = x.half()

        if not self.modiff_enabled or self.is_first_step:
            # ---- First step (t = T) ----
            if self.use_int8_gemm:
                o_hat = _compiled_first_step(x_f16, self.w_int8_t, self.w_scale, self.bias_fp32)
            else:
                o_hat = self._fp16_conv(x_f16)
            self.a_hat_cache = x_f16
            self.o_hat_cache = o_hat.half()
            self.is_first_step = False
            return o_hat.to(x.dtype)

        else:
            # ---- Modulated step (t < T) — hot path ----
            # Collect stats BEFORE updating cache (delta = x - prev_a_hat)
            if self._stat_calls < 200:
                with torch.no_grad():
                    self._stat_full_rms  += float(x_f16.float().pow(2).mean().sqrt())
                    self._stat_delta_rms += float(
                        (x_f16 - self.a_hat_cache).float().pow(2).mean().sqrt()
                    )
                self._stat_calls += 1

            if self.use_int8_gemm:
                o_f16 = _compiled_modulated_step(
                    x_f16, self.a_hat_cache, self.o_hat_cache,
                    self.w_int8_t, self.w_scale, self.bias_fp32,
                )
            else:
                delta = x_f16.float() - self.a_hat_cache.float()
                o_f16 = (self._fp16_conv(delta.half()) + self.o_hat_cache.float()).half()

            self.a_hat_cache = x_f16
            self.o_hat_cache = o_f16
            return o_f16.to(x.dtype)

    def reset_cache(self):
        """Call between DDIM samples to restart the temporal chain."""
        self.is_first_step = True
        self.a_hat_cache   = None
        self.o_hat_cache   = None


# ---------------------------------------------------------------------------
# Patched AttentionBlock _forward method
# ---------------------------------------------------------------------------

def _modiff_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Replacement for AttentionBlock._forward that uses MoDiffConv1d for qkv / proj_out.
    The softmax attention remains in FP16, unchanged.
    """
    b, c, *spatial = x.shape
    x_flat = x.reshape(b, c, -1)            # [B, C, HW]

    # GroupNorm (unchanged, FP16 / FP32 mixed via autocast)
    norm_x = self.norm(x_flat)              # [B, C, HW]

    # MoDiff QKV projection — was: self.qkv(norm_x)
    qkv = self.qkv(norm_x)                  # [B, 3C, HW]

    # Softmax attention (FP16, unchanged)
    h = self.attention(qkv)                 # [B, C, HW]

    # MoDiff proj_out — was: self.proj_out(h)
    h = self.proj_out(h)                    # [B, C, HW]

    return (x_flat + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------------

def convert_attention_to_modiff(
    unet: nn.Module,
    act_bits: int = 8,
    verbose: bool = True,
) -> int:
    """
    Replace the qkv and proj_out Conv1d layers of every AttentionBlock in
    `unet` with MoDiffConv1dCUTLASS (preferred) or MoDiffConv1d (fallback),
    and patch AttentionBlock._forward to use _modiff_attention_forward.

    Backend selection:
        CUTLASS available → MoDiffConv1dCUTLASS  (2 kernel launches / call)
        torch._int_mm     → MoDiffConv1d          (~12 kernel launches / call)

    Returns the number of attention blocks converted.
    """
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    import types

    backend = "CUTLASS" if HAS_CUTLASS_ATTN else ("torch._int_mm" if _HAS_INT_MM else "FP16")
    if verbose:
        print(f"  [MoDiffAttn] backend={backend}")

    converted = 0
    for name, module in unet.named_modules():
        if not isinstance(module, AttentionBlock):
            continue
        if not (isinstance(module.qkv, nn.Conv1d) and isinstance(module.proj_out, nn.Conv1d)):
            if verbose:
                print(f"  [MoDiffAttn] Skipping {name}: qkv/proj_out are not Conv1d")
            continue

        channels = module.channels
        # Wrapping nn.Conv1d in a fresh nn.Module (both branches build a plain nn.Conv2d
        # internally, which defaults to CPU) leaves some buffers -- e.g. smooth_scale,
        # created via torch.ones(...) with no device= -- stranded on CPU even though the
        # weight/bias tensors cloned from the original conv1d land on CUDA. This is silent
        # during plain inference (every buffer actually read there is either cloned from a
        # CUDA source or created matching x.device at call time) and only surfaces once
        # calibration reads a CPU-defaulted buffer like smooth_scale against a CUDA tensor.
        target_device = module.qkv.weight.device
        if HAS_CUTLASS_ATTN:
            module.qkv      = MoDiffConv1dCUTLASS(module.qkv,      layer_name=f"{name}.qkv")
            module.proj_out = MoDiffConv1dCUTLASS(module.proj_out,  layer_name=f"{name}.proj_out")
        else:
            module.qkv      = MoDiffConv1d(module.qkv,      act_bits=act_bits, layer_name=f"{name}.qkv")
            module.proj_out = MoDiffConv1d(module.proj_out,  act_bits=act_bits, layer_name=f"{name}.proj_out")
        if target_device.type != 'cpu':
            module.qkv = module.qkv.to(target_device)
            module.proj_out = module.proj_out.to(target_device)

        module._forward = types.MethodType(_modiff_attention_forward, module)
        converted += 1
        if verbose:
            print(f"  ✓ MoDiff attention: {name}  (C={channels}, backend={backend})")

    return converted


def reset_attention_modiff(unet: nn.Module):
    """Reset all MoDiff attention caches (call between DDIM samples)."""
    for module in unet.modules():
        if isinstance(module, (MoDiffConv1d, MoDiffConv1dCUTLASS)):
            module.reset_cache()


def set_attention_modiff_enabled(unet: nn.Module, enabled: bool):
    """Enable or disable the MoDiff delta trick for all attention layers."""
    for module in unet.modules():
        if isinstance(module, (MoDiffConv1d, MoDiffConv1dCUTLASS)):
            module.modiff_enabled = enabled


def get_attention_modiff_stats(unet: nn.Module) -> dict:
    """
    Return a dict of {layer_name: delta_ratio} from MoDiffConv1d layers.
    (CUTLASS backend does not expose per-step stats; only _int_mm backend does.)
    """
    stats = {}
    for module in unet.modules():
        if isinstance(module, MoDiffConv1d) and module._stat_calls > 0:
            n = module._stat_calls
            full_rms  = module._stat_full_rms  / n
            delta_rms = module._stat_delta_rms / n
            ratio = delta_rms / max(full_rms, 1e-8)
            stats[module.layer_name] = {
                "full_rms":    full_rms,
                "delta_rms":   delta_rms,
                "delta_ratio": ratio,
                "calls":       n,
                "backend":     "torch._int_mm",
            }
        elif isinstance(module, MoDiffConv1dCUTLASS):
            stats[module.layer_name] = {
                "backend": "CUTLASS",
                "calls":   module._stat_calls,
            }
    return stats
