"""Generalized-K variant of docs/ahat_skip2_exact_2026-08-26/scripts/patch_skip2.py.

patch_skip2.py hardcodes K=2 (`is_skip_step = (self.step_count % 2) == 1`, one pending-code
buffer, probe_skip/probe_catchup). This drives the already-built, already-bit-exact-verified
`probe_window_step` kernel instead, which takes (position, is_last) and a [K-1, numel] pending
buffer, so any K works from one code path. K=1 degenerates to the standard write-every-step
kernel, giving an apples-to-apples in-harness baseline through the identical dispatch.

Set K via MODIFF_SKIPK_K. Same scope/fallback rules as patch_skip2.

Apply: import patch_skipk; patch_skipk.install(K)
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
_K = 2


def _in_scope(self, x):
    return (not self.delta_dynamic and x.dtype == torch.float16
            and x.size(1) <= 1024 and x.size(1) % 2 == 0
            and not getattr(i8opt, "_GN_GROUPMAJOR", False))


def _skipk_forward_gn_fused_modiff(self, x, gn_weight, gn_bias, num_groups, eps,
                                   mod_scale2d, mod_shift2d, residual=None):
    if not _in_scope(self, x):
        _stats["fallback_calls"] += 1
        return _ORIGINAL(self, x, gn_weight, gn_bias, num_groups, eps,
                         mod_scale2d, mod_shift2d, residual)
    # REFRESH STEPS MUST TAKE THE PRODUCTION PATH. _delta_gn_dynamic_args returns REAL absmax
    # reduction buffers (and sets _delta_seeded) on every step where _delta_should_refresh() is
    # true -- step_count = 1, 1+R, 1+2R, ... for R = MODIFF_DELTA_REFRESH (default 4) -- and the
    # production kernel then runs an extra reduction/publish pass. probe_window_step has no
    # equivalent parameters at all, so it silently implements only the NON-refresh branch. That is
    # also the only branch sweep_k.py ever compared against (it passed all-empty buffers), which is
    # why the isolated-kernel sweep reported bit-exact while a real run diverges.
    # Measured: at R=4 the window patch differs from production by 78/255 on 11.8% of pixels and is
    # not even run-to-run stable; at R=1000 (only step 1 refreshes) it is bit-exact on every
    # comparison. So: delegate refresh steps, and window only the runs between them.
    R = max(1, int(getattr(self, "delta_refresh", 1)))
    phase = (self.step_count % R) if R > 1 else 1   # step_count is PRE-increment here
    if R > 1 and phase == 0:
        _stats["patched_calls"] -= 1
        _stats["refresh_delegated"] = _stats.get("refresh_delegated", 0) + 1
        return _ORIGINAL(self, x, gn_weight, gn_bias, num_groups, eps,
                         mod_scale2d, mod_shift2d, residual)
    _stats["patched_calls"] += 1

    self.step_count += 1
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    self._ensure_state_buffers(x)
    d_scale, d_alpha = self._delta_scale_args(x.device)

    N, C, H, W = x.shape
    G = num_groups
    ne, ss = N * C * H * W, C * H * W

    stats_key = (N, G, x.device)
    if getattr(self, "_skipk_stats_key", None) != stats_key:
        self._skipk_mean = torch.empty(N * G, device=x.device, dtype=torch.float32)
        self._skipk_inv_std = torch.empty(N * G, device=x.device, dtype=torch.float32)
        self._skipk_stats_key = stats_key
    mean, inv_std = self._skipk_mean, self._skipk_inv_std

    if getattr(self, "_skipk_empty16", None) is None or self._skipk_empty16.device != x.device:
        self._skipk_empty16 = torch.empty(0, device=x.device, dtype=torch.float16)
    empty16 = self._skipk_empty16
    ms = mod_scale2d if mod_scale2d.numel() > 0 else empty16
    sh = mod_shift2d if mod_shift2d.numel() > 0 else empty16
    if not self._smooth_is_identity and self._smooth_inv_flat.numel() > 0:
        if getattr(self, "_skipk_smooth_inv_half", None) is None:
            self._skipk_smooth_inv_half = self._smooth_inv_flat.half()
        si = self._skipk_smooth_inv_half
    else:
        si = empty16

    # Per-(shape) window state: the pending-code ring and the position counter must both reset
    # when the shape changes, or a stale buffer would be read as if it held this shape's codes.
    buf_key = (N, C, H, W, x.device, _K)
    if getattr(self, "_skipk_buf_key", None) != buf_key:
        slots = max(_K - 1, 1)
        self._skipk_pending = torch.zeros(slots, ne, device=x.device, dtype=torch.int8)
        self._skipk_pending_inv = torch.zeros(slots, device=x.device, dtype=torch.float32)
        self._skipk_yq = torch.empty(N, C, H, W, device=x.device, dtype=torch.int8).to(
            memory_format=torch.channels_last)
        self._skipk_buf_key = buf_key
    # Derive the window position from step_count rather than an independent counter. An independent
    # counter survives reset_modiff_state (which zeroes a_hat and step_count), so a generation that
    # ended mid-window left the next one reconstructing from a ZEROED a_hat plus the previous run's
    # pending codes -- measured catastrophic (255/255 on 99.7% of pixels). step_count is reset with
    # a_hat, so tying the window to it makes that desync unrepresentable. The isolated-kernel
    # K-sweep could not see this class of bug: it only ran exactly K chained calls, never a partial
    # window or a state reset.
    # Window only the run of non-refresh steps between two refresh steps. `phase` (1..R-1) is this
    # step's offset inside that run, so pos_in_run = phase-1 and the window must CLOSE (write a_hat)
    # at phase == R-1, before the next refresh step delegates to production and reads a_hat.
    # K_eff is therefore capped by the run length: with the default R=4 the longest usable window
    # is 3, which is also where the isolated-kernel sweep put its optimum.
    K_eff = max(1, min(_K, R - 1)) if R > 1 else _K
    pos_in_run = phase - 1
    position = pos_in_run % K_eff
    is_last = (position == K_eff - 1) or (R > 1 and phase == R - 1)

    probe.stats_launch(x, mean, inv_std, C, G, H * W, eps)
    x_int8 = self._skipk_yq
    probe.probe_window_step_launch(x, self.a_hat_cache, x_int8, gn_weight, gn_bias, ms, sh, si,
                                   mean, inv_std, d_scale,
                                   self._skipk_pending, self._skipk_pending_inv,
                                   ne, position, is_last, C, G, ss, ne, True, self._delta_a4)
    if not is_last:
        self._skipk_pending_inv[position].copy_(d_alpha.view(-1)[0])

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


def install(K=None):
    global _K
    _K = int(K if K is not None else os.environ.get("MODIFF_SKIPK_K", "2"))
    _Cls.forward_gn_fused_modiff = _skipk_forward_gn_fused_modiff
    _stats["patched_calls"] = 0
    _stats["fallback_calls"] = 0
    _stats["refresh_delegated"] = 0
    return _K


def uninstall():
    _Cls.forward_gn_fused_modiff = _ORIGINAL


def report():
    print(f"skipK(K={_K}) patch: {_stats['patched_calls']} patched, "
          f"{_stats['fallback_calls']} fell back, "
          f"{_stats.get('refresh_delegated',0)} refresh-delegated")
