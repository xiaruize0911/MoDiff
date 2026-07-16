"""Weight+activation int quantized Linear (W8A8 / W4A4), static scales, for the
UNet Linear-equivalent layers. One strategy parameterized by bit-width:
per-output-channel symmetric weights + static per-tensor symmetric activations,
running the custom int tensor-core GEMM (`gemm_w8a8` / `gemm_w4a4`).

Weights are quantized once at construction (static). The activation scale is
static (set by calibration via `set_a_scale`); if unset, a dynamic per-tensor
absmax scale is used as a fallback (non-CUDA-graph).

Eligibility (else keep fp16, handled by the converter): out%64==0 and
in%32==0 (int8) / in%64==0 (int4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import modiff_cutlass as _mc
    _HAS = hasattr(_mc, "gemm_w8a8") and hasattr(_mc, "gemm_w4a4")
except Exception:
    _mc = None
    _HAS = False

# Optional: route eligible W8A8 through AWQ's tuned kernel (MODIFF_WXAX_USE_AWQ=1) to
# measure the achievable e2e ceiling. AWQ needs N%128 (large-M CTA_N).
import os as _os
_USE_AWQ = _os.environ.get("MODIFF_WXAX_USE_AWQ") == "1"
_awq = None
if _USE_AWQ:
    try:
        import sys as _sys
        _sys.path.insert(0, "/workspace/llm-awq/awq/kernels")
        import awq_inference_engine as _awq
    except Exception:
        _awq = None


def _pack4(q):  # q int8 [...,K] in [-7,7] -> [...,K/2] int8 (2 int4/byte, low nibble = even)
    q = q.to(torch.int32)
    lo = q[..., 0::2] & 0xF
    hi = q[..., 1::2] & 0xF
    return ((lo | (hi << 4)).to(torch.int8)).contiguous()


def _eligible(in_f, out_f, bits):
    return _HAS and (out_f % 64 == 0) and (in_f % (64 if bits == 4 else 32) == 0)


class QuantLinearWxAx(nn.Module):
    def __init__(self, lin: nn.Linear, bits: int):
        super().__init__()
        assert bits in (4, 8) and _eligible(lin.in_features, lin.out_features, bits)
        self.bits = bits
        self.Q = 127 if bits == 8 else 7
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        W = lin.weight.detach().float()                       # [N,K]
        s = (W.abs().amax(1).clamp_min(1e-8) / self.Q)        # per-output-channel [N]
        Wq = torch.round(W / s.unsqueeze(1)).clamp(-self.Q, self.Q).to(torch.int8)
        self.register_buffer("w_scale", s.to(torch.float32).contiguous())
        self.register_buffer("qweight", _pack4(Wq) if bits == 4 else Wq.contiguous())
        self.register_buffer("bias", lin.bias.detach().half().contiguous() if lin.bias is not None else None)
        self.a_scale = None                                    # static activation scale (calibrated)
        self._calib = False
        self._amax = 0.0

    def set_a_scale(self, s):
        self.a_scale = float(s)

    def forward(self, x):
        orig = x.shape
        xf = x.reshape(-1, self.in_features)
        if self._calib:                                          # record activation absmax (static calib)
            self._amax = max(self._amax, float(xf.abs().max()))
        if self.a_scale is not None:
            a_scale = self.a_scale
        else:
            a_scale = (xf.abs().max().item() / self.Q) or 1e-8   # dynamic fallback (host sync)
        xf = xf.half().contiguous()
        if self.bits == 8:
            xq = _mc.quantize_act_int8(xf, a_scale)                  # fused fp16->int8
            if _awq is not None and self.out_features % 128 == 0:
                M = xq.shape[0]
                asc = torch.full((M,), a_scale, device=xq.device, dtype=torch.float16)
                out = torch.empty(M, self.out_features, device=xq.device, dtype=torch.float16)
                _awq.w8a8_gemm_forward_cuda(xq, self.qweight, self.w_scale.to(torch.float16), asc, out)
            else:
                out = _mc.gemm_w8a8(xq, self.qweight, self.w_scale, a_scale)
        else:
            xq = _mc.quantize_act_int4_pack(xf, a_scale)             # fused fp16->packed int4
            out = _mc.gemm_w4a4(xq, self.qweight, self.w_scale, a_scale, self.in_features)
        out = out.reshape(*orig[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


def set_wxax_calibrating(model, flag):
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx):
            m._calib = flag
            if flag:
                m._amax = 0.0


def finalize_wxax_ascale(model):
    """Set each layer's static activation scale from the calibrated absmax."""
    n = 0
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx) and m._amax > 0:
            m.set_a_scale(m._amax / m.Q)
            m._calib = False
            n += 1
    return n


def convert_linears_to_wxax(module, bits, verbose=False, _prefix=""):
    """Recursively replace eligible nn.Linear with QuantLinearWxAx(bits). Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if _eligible(child.in_features, child.out_features, bits):
                setattr(module, name, QuantLinearWxAx(child, bits).to(child.weight.device))
                n += 1
                if verbose:
                    print(f"  wxax({bits}): {_prefix}{name} {child.in_features}->{child.out_features}")
            elif verbose:
                print(f"  skip(fp16): {_prefix}{name} {child.in_features}->{child.out_features}")
        else:
            n += convert_linears_to_wxax(child, bits, verbose, _prefix + name + ".")
    return n
