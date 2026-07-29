"""Per-STEP, per-SHAPE benchmark of the AttentionBlock, in the exact kernel groupings production uses.

The layer is six structural steps (see ATTENTION_ANALYSIS.md §1):
  S1 GroupNorm(32 groups, no SiLU)        S4 softmax(QKᵀ/sqrt(hd))·V
  S2 qkv projection  [NT,C]@[C,3C]        S5 out projection  [NT,C]@[C,C]
  S3 split/head-transpose                 S6 residual add
Every mode implements the same six, but groups them into a DIFFERENT set of kernels, and the
quantized modes add quantize/dequantize work that is not in the six at all. Timing the six
separately would therefore misrepresent every mode: the whole point of the fusions is that S1+S2
(or S5+S6) is one launch, so a "per-step" number only means something if each measured item is a
real launch group.

So each row here is ONE production launch group, labelled with the steps it covers, and the fp16
comparison is made group-to-group over the SAME steps. Rows are measured with synthetic tensors at
the real shapes (from configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml: model_channels=192,
channel_mult=[1,2,2,4,4], num_heads=8, latent 32x32 -> hd=C/8, T=H*W), which is what the
model-level scripts do too; the checkpoint in this tree has an empty state_dict, and these kernels
have no data-dependent control flow, so weights would not change any timing.

Weight/activation formats come from real QuantLinearWxAx instances, not hand-rolled tensors, so the
AWQ N/K padding, the per-output-channel weight scales and the packed int4 layout are exactly what
the model runs -- including the C=192 int4 case where _awqt_K (256) != in_features (192), which now
stays on the fused GN->pack path via k_pad (the `int4_old` rows show what the previous fallback cost).
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import modiff_cutlass as mc
from integration.kernels.wxax_linear import QuantLinearWxAx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "attn_step_bench.json")
DEV = "cuda"
BATCH, HEADS, GROUPS = 128, 8, 32
FUSE_SHIFT, FUSE_TILE_M = 16.0, 128
BLOCKS = [(192, 32, 32, 5), (384, 16, 16, 5), (384, 8, 8, 5), (768, 4, 4, 5), (768, 2, 2, 1)]


def bench(fn, it=25, reps=5):
    """Median over reps windows. Returns us/call, or the exception string."""
    try:
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:90]}"
    outs = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        outs.append(s.elapsed_time(e) / it * 1e3)
    return statistics.median(outs), None


def eligible_flash(T, hd):
    return ((hd + 31) // 32) * 32 <= 64 and T % 64 == 0 and hd % 8 == 0


def make_quant_linear(in_f, out_f, bits):
    lin = nn.Linear(in_f, out_f).to(DEV).half()
    q = QuantLinearWxAx(lin, bits=bits, modiff=False).to(DEV)
    q.set_a_scale(0.02)
    q._use_bias_res = True
    q._calib = False
    return q


def run_shape(C, Hh, Ww, count):
    T, hd = Hh * Ww, C // HEADS
    N = BATCH
    hd_pad = ((hd + 31) // 32) * 32
    scale = 1.0 / (hd ** 0.5)
    fusable_gn_qkv = (T % FUSE_TILE_M == 0) and (C % 8 == 0)
    flash_ok = eligible_flash(T, hd)
    rows = []

    def add(mode, steps, label, fn, it=25, **extra):
        us, err = bench(fn, it=it)
        rows.append(dict(mode=mode, steps=steps, label=label, us=us, error=err, **extra))
        tag = f"{us:9.1f} us" if us is not None else f"ERR {err}"
        print(f"    {mode:6s} {steps:9s} {label:52s} {tag}")

    # ---------------- shared inputs ----------------
    x_img = torch.randn(N, C, Hh, Ww, device=DEV, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    x_tok = x_img.permute(0, 2, 3, 1).reshape(N, T, C)
    gw = torch.randn(C, device=DEV, dtype=torch.float16)
    gb = torch.randn(C, device=DEV, dtype=torch.float16)
    gn = nn.GroupNorm(GROUPS, C).to(DEV).half()
    qkv_f = nn.Linear(C, 3 * C).to(DEV).half()
    proj_f = nn.Linear(C, C).to(DEV).half()
    empty = x_img.new_empty(0)

    # ================= fp16 =================
    # S1+S2: fused GN->qkv (one CUTLASS kernel) where T%128==0, else GN kernel + cuBLAS GEMM.
    if fusable_gn_qkv:
        Wf = (qkv_f.weight.detach().float() * gw.float()[None, :])
        cw = Wf.to(torch.float16).view(3 * C, 1, 1, C).contiguous()
        eb = (qkv_f.bias.detach().float() + qkv_f.weight.detach().float() @ gb.float()
              - FUSE_SHIFT * Wf.sum(1)).to(torch.float16).contiguous()
        add("fp16", "S1+S2", "fused_gn_qkv (GN folded into qkv conv, 1 kernel)",
            lambda: mc.fused_gn_qkv(x_img, cw, eb, GROUPS, 1e-5, FUSE_SHIFT), it=15)
    else:
        add("fp16", "S1", "group_norm_silu_nhwc (no SiLU)",
            lambda: mc.group_norm_silu_nhwc(x_img, gw, gb, GROUPS, 1e-5, False,
                                            empty, empty), it=15)
        add("fp16", "S2", "qkv GEMM fp16 (cuBLAS F.linear)",
            lambda: F.linear(x_tok, qkv_f.weight, qkv_f.bias), it=15)

    qkv_pk = torch.randn(N, T, HEADS, 3, hd, device=DEV, dtype=torch.float16).contiguous()
    q = qkv_pk[:, :, :, 0].permute(0, 2, 1, 3).contiguous()
    k = qkv_pk[:, :, :, 1].permute(0, 2, 1, 3).contiguous()
    v = qkv_pk[:, :, :, 2].permute(0, 2, 1, 3).contiguous()
    add("fp16", "S3", "q/k/v transpose materialize (only if not read packed)",
        lambda: qkv_pk[:, :, :, 0].permute(0, 2, 1, 3).contiguous())
    add("fp16", "S4", "PyTorch fp16 SDPA (flash)",
        lambda: F.scaled_dot_product_attention(q, k, v, scale=scale))
    fp16_sdpa_us = rows[-1]["us"]     # reused below: the quantized modes fall back to this when
                                      # our flash kernel cannot serve the shape (hd=96 blocks)
    a = torch.randn(N, HEADS, T, hd, device=DEV, dtype=torch.float16)
    add("fp16", "S5a", "attn-out transpose head->token major (copy)",
        lambda: a.transpose(1, 2).reshape(N, T, C).contiguous())
    at = a.transpose(1, 2).reshape(N, T, C).contiguous()
    add("fp16", "S5b", "proj GEMM fp16 (cuBLAS F.linear)",
        lambda: F.linear(at, proj_f.weight, proj_f.bias))
    pr = F.linear(at, proj_f.weight, proj_f.bias)
    add("fp16", "S6", "residual add (separate elementwise)", lambda: pr + x_tok)

    # ================= int8 / int4 =================
    for bits in (8, 4):
        mode = f"int{bits}"
        qkv_q = make_quant_linear(C, 3 * C, bits)
        proj_q = make_quant_linear(C, C, bits)
        inv_s = torch.tensor([1.0 / 0.02], device=DEV, dtype=torch.float32)
        # Production gate (token_major_attention._qkv_from_gn): int8 needs K == in_features because
        # its GN kernel has no k_pad; int4 handles a padded K inside the pack kernel, so it is always
        # fused now. Before k_pad existed, int4 with K_pad > C fell back to
        # group_norm_silu_nhwc + F.pad + a standalone quantize -- kept below as `int4_unfused` so the
        # report can still show what that cost.
        k_ok = (bits == 4) or (qkv_q._awqt_K == qkv_q.in_features)
        kp = qkv_q._awqt_K

        # ---- S1+S2(+activation quantize): GN emits int8/int4 directly, then the int GEMM with
        # bias folded into its epilogue. Two launches for three jobs.
        if k_ok:
            if bits == 8:
                add(mode, "S1+quant", "GN -> int8 in one kernel (quantize fused into GN)",
                    lambda: mc.group_norm_silu_quantize_nhwc(x_img, gw, gb, GROUPS, 1e-5, False,
                                                             inv_s, empty, empty, empty), it=15)
                xq_img = mc.group_norm_silu_quantize_nhwc(x_img, gw, gb, GROUPS, 1e-5, False,
                                                          inv_s, empty, empty, empty)
                xq = xq_img.permute(0, 2, 3, 1).reshape(N * T, C)
                add(mode, "S2", "qkv GEMM w8a8 (+bias in epilogue)",
                    lambda: mc.gemm_w8a8_awq_bias_res(xq, qkv_q.qweight, qkv_q.w_scale, 0.02,
                                                      qkv_q.out_features, qkv_q.bias, empty), it=15)
            else:
                # k_pad = the GEMM's padded K, so a C not on the K tile (192 -> 256) stays fused.
                add(mode, "S1+quant", f"GN -> int4 in one kernel, k_pad {C}->{kp}"
                                      f"{' (no pad needed)' if kp == C else ''}",
                    lambda: mc.group_norm_silu_quantize_pack_nhwc(x_img, gw, gb, GROUPS, 1e-5,
                                                                  False, inv_s, empty, empty,
                                                                  empty, kp), it=15)
                xq_img = mc.group_norm_silu_quantize_pack_nhwc(x_img, gw, gb, GROUPS, 1e-5, False,
                                                               inv_s, empty, empty, empty, kp)
                xq = xq_img.reshape(N * T, kp // 2)
                add(mode, "S2", "qkv GEMM w4a4 (+bias in epilogue)",
                    lambda: mc.gemm_w4a4_awq_bias_res(xq, qkv_q.qweight, qkv_q.w_scale, 0.02,
                                                      kp, qkv_q.out_features,
                                                      qkv_q.bias, empty), it=15)
                if kp != C:
                    # The pre-k_pad fallback, for the report's before/after. Not a production path.
                    def _unfused():
                        xn = mc.group_norm_silu_nhwc(x_img, gw, gb, GROUPS, 1e-5, False, empty, empty)
                        t = xn.permute(0, 2, 3, 1).reshape(N * T, C)
                        return mc.quantize_act_int4_pack(F.pad(t, (0, kp - C)).contiguous(), 0.02)
                    add(f"{mode}_old", "S1+quant", "WAS: GN fp16 + F.pad + standalone quantize "
                                                  "(3 kernels, pre-k_pad)", _unfused, it=15)

        # ---- S3+S4: quantize Q/K/V then our flash, or the packed variant that folds the
        # quantize into flash's smem staging. Also the fp16 SDPA the gate may pick instead.
        if flash_ok:
            # hp_qk is the Q/K pad (int4 packs to 64); hp_av is the V pad. sv_vec is indexed by
            # hp_av ONLY -- the kernel TORCH_CHECKs sv_vec.numel() == hp_av, so it must not be
            # sized by max(hp_qk, hp_av).
            hdp_qk = hd_pad if bits == 8 else 64
            svv = torch.full((hd_pad,), 0.01, device=DEV)
            add(mode, "S3+quant", f"quantize_attn_qkv_packed_static (Q/K int{bits}, V int8, +transpose)",
                lambda: mc.quantize_attn_qkv_packed_static(qkv_pk, HEADS, T, hd, hdp_qk,
                                                           hd_pad, bits, 0.02, 0.02, svv), it=15)
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_packed_static(
                qkv_pk, HEADS, T, hd, hdp_qk, hd_pad, bits, 0.02, 0.02, svv)
            qi = qi.view(N, HEADS, T, -1); ki = ki.view(N, HEADS, T, -1)
            vt = vt.view(N, HEADS, hd_pad, T)
            sq = sq.view(N, HEADS, T).contiguous(); sk = sk.view(N, HEADS, T).contiguous()
            sv = sv[..., :hd].contiguous().view(N, HEADS, hd)
            if bits == 8:
                add(mode, "S4", "flash_attn_int8_vt (QKᵀ+softmax+AV, scores in SRAM)",
                    lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, scale))
                sv_hd = torch.full((hd,), 0.01, device=DEV)
                add(mode, "S3+S4", "flash_attn_int8_packed_vt (quantize folded into flash)",
                    lambda: mc.flash_attn_int8_packed_vt(qkv_pk, sv_hd, hd_pad, 0.02, 0.02, scale))
                add(mode, "S4+S5q", "flash_attn_int8_vt_qout (emits proj-quantized int8 out)",
                    lambda: mc.flash_attn_int8_vt_qout(qi, ki, vt, sq, sk, sv, scale, 0.02))
            else:
                add(mode, "S4", "flash_attn_int4_vt (int4 Q/K, int8 V/P)",
                    lambda: mc.flash_attn_int4_vt(qi, ki, vt, sq, sk, sv, 64, scale))
                add(mode, "S4+S5q", "flash_attn_int4_vt_qout (emits proj-quantized int4 out)",
                    lambda: mc.flash_attn_int4_vt_qout(qi, ki, vt, sq, sk, sv, 64, scale,
                                                       0.02, proj_q._awqt_K))
        else:
            # Our flash cannot serve hd>48, so these blocks run the fp16 SDPA -- they still PAY for
            # S4, just at fp16 cost. Recording it as the measured fp16 time (rather than None) keeps
            # the per-mode totals honest; leaving it out silently credited the quantized modes with
            # free attention on 6 of the 21 blocks.
            rows.append(dict(mode=mode, steps="S4", label="INELIGIBLE for our flash "
                             f"(hd={hd}>48 or T={T}%64!=0) -> falls back to fp16 SDPA",
                             us=fp16_sdpa_us, error=None, is_fp16_fallback=True))
            print(f"    {mode:6s} S4        ineligible (hd={hd}, T={T}) -> fp16 SDPA "
                  f"{fp16_sdpa_us:.1f} us")

        # ---- S5+S6: transpose+quantize in one kernel, then the int GEMM with bias AND the skip
        # residual folded into its store epilogue. Two launches for four jobs.
        if bits == 8:
            add(mode, "S5a+quant", "quantize_attn_out_int8 (transpose + quantize, 1 kernel)",
                lambda: mc.quantize_attn_out_int8(a, 0.02))
            xo = mc.quantize_attn_out_int8(a, 0.02)
            res = x_tok.reshape(N * T, C).contiguous()
            add(mode, "S5b+S6", "proj GEMM w8a8 (+bias +residual in epilogue)",
                lambda: mc.gemm_w8a8_awq_bias_res(xo, proj_q.qweight, proj_q.w_scale, 0.02,
                                                  proj_q.out_features, proj_q.bias, res))
        else:
            add(mode, "S5a+quant", "quantize_attn_out_int4_pack (transpose + quantize + pack)",
                lambda: mc.quantize_attn_out_int4_pack(a, 0.02, proj_q._awqt_K))
            xo = mc.quantize_attn_out_int4_pack(a, 0.02, proj_q._awqt_K)
            res = x_tok.reshape(N * T, C).contiguous()
            add(mode, "S5b+S6", "proj GEMM w4a4 (+bias +residual in epilogue)",
                lambda: mc.gemm_w4a4_awq_bias_res(xo, proj_q.qweight, proj_q.w_scale, 0.02,
                                                  proj_q._awqt_K, proj_q.out_features,
                                                  proj_q.bias, res))
        del qkv_q, proj_q
        torch.cuda.empty_cache()

    del x_img, x_tok, qkv_pk, q, k, v, a, at, pr
    torch.cuda.empty_cache()
    return dict(C=C, HW=f"{Hh}x{Ww}", T=T, hd=hd, hd_pad=hd_pad, count=count,
                gn_qkv_fusable=fusable_gn_qkv, flash_eligible=flash_ok,
                flops_qkv_g=round(2 * BATCH * T * C * 3 * C / 1e9, 2),
                flops_attn_g=round(4 * BATCH * HEADS * T * T * hd / 1e9, 2),
                flops_proj_g=round(2 * BATCH * T * C * C / 1e9, 2),
                rows=rows)


def main():
    bn = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    out = {"batch": BATCH, "heads": HEADS, "groups": GROUPS, "shapes": []}
    for C, Hh, Ww, cnt in BLOCKS:
        print(f"\n=== C={C} {Hh}x{Ww}  T={Hh*Ww}  hd={C//HEADS}  x{cnt} instances ===")
        out["shapes"].append(run_shape(C, Hh, Ww, cnt))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
