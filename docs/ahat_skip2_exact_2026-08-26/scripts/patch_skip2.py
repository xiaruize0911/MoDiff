"""Wires the K=2 deferred-write a_hat scheme into the real generation pipeline via a monkeypatch
of OptimizedInt8Conv2d.forward_gn_fused_modiff -- the fused GN+SiLU+delta-quantize entry point
fused_resblock.py actually dispatches to for the production int8 path.

Only the "Step1" computation (silu(gn(x)) -> delta-quantize -> a_hat update) is replaced, by
calling into the ahat_skip2_probe CUDA extension (validated against real production output with
modulation active in verify_vs_production.py) instead of modiff_cutlass.group_norm_silu_delta_quantize_nhwc.
Everything after Step1 (the o_hat conv, residual fusion) is untouched -- it only ever consumes the
returned code tensor, which is bit-identical to what the real kernel would have produced (that IS
the whole point of the design).

Scope, falls back to the ORIGINAL method outside of it:
  - static delta mode only (self.delta_dynamic == False)
  - x.dtype == float16
  - C <= 1024 and C % 2 == 0 (the K=1 chanmajor stats path; excludes the two C=1152/1536 decoder
    concat blocks)
  - the experimental group-major kernel switch (_GN_GROUPMAJOR) off (the shipped default)

Apply with: import patch_skip2; patch_skip2.install()
Remove with: patch_skip2.uninstall()
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")
import ahat_skip2_probe as probe  # noqa: E402
import torch  # noqa: E402

import integration.kernels.int8_optimized as i8opt  # noqa: E402

_Cls = i8opt.OptimizedInt8Conv2d
_ORIGINAL = _Cls.forward_gn_fused_modiff
_stats = {"patched_calls": 0, "fallback_calls": 0}


def _in_scope(self, x):
    return (not self.delta_dynamic and x.dtype == torch.float16
            and x.size(1) <= 1024 and x.size(1) % 2 == 0
            and not getattr(i8opt, "_GN_GROUPMAJOR", False))


def _skip2_forward_gn_fused_modiff(self, x, gn_weight, gn_bias, num_groups, eps,
                                   mod_scale2d, mod_shift2d, residual=None):
    if not _in_scope(self, x):
        _stats["fallback_calls"] += 1
        return _ORIGINAL(self, x, gn_weight, gn_bias, num_groups, eps,
                         mod_scale2d, mod_shift2d, residual)
    _stats["patched_calls"] += 1

    self.step_count += 1
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    self._ensure_state_buffers(x)
    d_scale, d_alpha = self._delta_scale_args(x.device)  # static mode: table slices, no sync

    N, C, H, W = x.shape
    G = num_groups

    # Cache all per-layer scratch buffers on self, sized once per (N,G)/(N,C,H,W) -- allocating
    # fresh tensors every call was measured to swamp the kernel-level saving at small batch/step
    # counts (the whole point of this patch is a ~0.5 ms/step KERNEL saving; a fresh cudaMalloc
    # path per call costs far more than that).
    stats_key = (N, G, x.device)
    if getattr(self, "_skip2_stats_key", None) != stats_key:
        self._skip2_mean = torch.empty(N * G, device=x.device, dtype=torch.float32)
        self._skip2_inv_std = torch.empty(N * G, device=x.device, dtype=torch.float32)
        self._skip2_stats_key = stats_key
    mean, inv_std = self._skip2_mean, self._skip2_inv_std
    probe.stats_launch(x, mean, inv_std, C, G, H * W, eps)

    if getattr(self, "_skip2_empty16", None) is None or self._skip2_empty16.device != x.device:
        self._skip2_empty16 = torch.empty(0, device=x.device, dtype=torch.float16)
    empty16 = self._skip2_empty16
    ms = mod_scale2d if mod_scale2d.numel() > 0 else empty16
    sh = mod_shift2d if mod_shift2d.numel() > 0 else empty16
    if not self._smooth_is_identity and self._smooth_inv_flat.numel() > 0:
        if getattr(self, "_skip2_smooth_inv_half", None) is None:
            self._skip2_smooth_inv_half = self._smooth_inv_flat.half()
        si = self._skip2_smooth_inv_half
    else:
        si = empty16

    is_skip_step = (self.step_count % 2) == 1
    ne, ss = N * C * H * W, C * H * W
    if is_skip_step:
        pcode_key = (N, C, H, W, x.device)
        if getattr(self, "_skip2_pcode_key", None) != pcode_key:
            self._skip2_pending_code = torch.empty(N, C, H, W, device=x.device, dtype=torch.int8).to(
                memory_format=torch.channels_last)
            self._skip2_pcode_key = pcode_key
            self._skip2_pending_inv_scale = torch.empty(1, device=x.device, dtype=torch.float32)
        x_int8 = self._skip2_pending_code
        probe.probe_skip_launch(x, self.a_hat_cache, x_int8, gn_weight, gn_bias, ms, sh, si,
                                mean, inv_std, d_scale, C, G, ss, ne, True, self._delta_a4)
        self._skip2_pending_inv_scale.copy_(d_alpha)
        self._skip2_has_pending = True
    else:
        if not getattr(self, "_skip2_has_pending", False) or self._skip2_pcode_key[:4] != (N, C, H, W):
            # No valid pending code (e.g. this is the very first modulated call and somehow hit
            # the even branch, or shape changed) -- fail safe to the real kernel rather than
            # silently computing wrong output.
            _stats["fallback_calls"] += 1
            self.step_count -= 1  # the original method also increments; undo our increment first
            return _ORIGINAL(self, x, gn_weight, gn_bias, num_groups, eps,
                             mod_scale2d, mod_shift2d, residual)
        x_int8 = self._skip2_pending_code  # overwritten in place by probe_catchup_launch
        probe.probe_catchup_launch(x, self.a_hat_cache, x_int8, gn_weight, gn_bias, ms, sh, si,
                                   mean, inv_std, d_scale, self._skip2_pending_inv_scale,
                                   C, G, ss, ne, True, self._delta_a4)
        self._skip2_has_pending = False

    if self._delta_calib:
        self._observe_delta_codes(x_int8)

    modiff_cutlass = i8opt.modiff_cutlass
    if residual is not None:
        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
        out = torch.empty_like(self.o_hat_cache)
        modiff_cutlass.conv2d_int8_evt_o_hat_residual(
            x_int8, self.weight_int8, d_alpha,
            self.weight_scale_channel.view(-1), self.o_hat_cache, residual, out,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        return out

    (modiff_cutlass.conv2d_int8_evt_o_hat if self.o_hat_cache.dtype == torch.float16
     else modiff_cutlass.conv2d_int8_fprop_o_hat)(
        x_int8, self.weight_int8, d_alpha,
        self.weight_scale_channel.view(-1), self.o_hat_cache,
        self.stride[0], self.stride[1], self.padding[0], self.padding[1],
        self.dilation[0], self.dilation[1])
    return self._module_output()


def install():
    _Cls.forward_gn_fused_modiff = _skip2_forward_gn_fused_modiff
    _stats["patched_calls"] = 0
    _stats["fallback_calls"] = 0


def uninstall():
    _Cls.forward_gn_fused_modiff = _ORIGINAL


def report():
    print(f"skip2 patch: {_stats['patched_calls']} calls patched, "
          f"{_stats['fallback_calls']} fell back to the original kernel")
