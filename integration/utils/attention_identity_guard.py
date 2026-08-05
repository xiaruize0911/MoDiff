"""Guard against correctness checks that cannot fail in this tree.

The problem
-----------
`zero_module()` zero-initialises the last layer of several LDM blocks so training starts as the
identity, and `models/ldm/lsun_churches256/model.ckpt` here is an 856-byte stub whose
`state_dict` has 0 entries, loaded with `strict=False`. Nothing ever overwrites those zeros, so
they stay zero at inference. Two consequences, measured 2026-08-03:

* `AttentionBlock.proj_out` is a zero_module (ldm/modules/diffusionmodules/openaimodel.py:345),
  so every attention block computes `x + proj_out(attention(norm(x))) == x` -- a bit-exact
  identity. 21/21 blocks, in fp16, int8_baseline and int4_baseline alike.
* `UNetModel.out[-1]` is a zero_module too (openaimodel.py:745), so the UNet's epsilon
  prediction is **identically zero for every input**. The sampled latent is therefore a
  deterministic function of the initial noise and the DDIM schedule alone, with no dependence on
  anything inside the UNet.

The second point is the severe one: it makes EVERY latent-level check vacuous, for every change
anywhere in the UNet -- attention, convolution, GroupNorm, quantization. Verified by forcing all
21 attention blocks to return a constant during sampling: `forward` fired 420 times and the
latent was bit-identical (docs/gn_qkv_fusion_2026-08-03/FINDINGS.md section 5).

What to use instead
-------------------
* Kernel-level checks against an fp32 reference computed from the SAME quantized codes the
  kernel consumes, on synthetic tensors at production shapes -- no checkpoint, no proj_out, no
  UNet output layer. `docs/final_report_2026-07-28/scripts/qattn_correctness.py` and
  `int4_fused_routes_check.py` are the models to copy.
* For layer- or model-level EQUIVALENCE checks (route A must agree with route B), call
  `activate_zeroed_modules(model)` first: it restores the default initialisation `zero_module`
  overwrote, which makes the whole pipeline observable again.
* Either way call `assert_unet_output_observable()` (behavioural, cannot be fooled) or
  `assert_attention_observable()` (structural, cheap) so a regression here fails loudly instead
  of silently passing.

Two cautions
------------
* Activating these layers changes the model, so a golden captured with them active is not
  comparable to one captured without. Key any cached reference on that flag.
* Activation makes an EQUIVALENCE check meaningful. It does not make a latent comparison a
  QUALITY measurement -- every weight in this tree is random, so latent relative-L2 against fp16
  measures agreement between two numeric routes, never image quality.
"""

import math
import zlib

import torch

ATTENTION_CLASSES = ("AttentionBlock", "TokenMajorAttentionBlock",
                     "QuantizedStandardAttentionBlock")


class NotObservable(RuntimeError):
    """What the caller is about to compare cannot be influenced by what it is testing."""


# Kept as an alias: earlier drafts of this module raised this name.
AttentionNotObservable = NotObservable


# ---------------------------------------------------------------- discovery


def attention_blocks(model):
    """[(qualified_name, module)] for every attention block in `model`."""
    return [(name, mod) for name, mod in model.named_modules()
            if type(mod).__name__ in ATTENTION_CLASSES]


def _effective_weight_is_zero(mod):
    """True when this module annihilates any input, i.e. its weight is identically zero.

    Handles both dense modules (`weight`) and QuantLinearWxAx (`qweight` * `w_scale`, where the
    logical block is N/K zero-padded to the AWQ tile, so only the logical part is examined).
    """
    qweight = getattr(mod, "qweight", None)
    if qweight is not None:
        if not bool((qweight != 0).any()):
            return True
        w_scale = getattr(mod, "w_scale", None)
        if w_scale is None:
            return False
        n = getattr(mod, "out_features", w_scale.numel())
        return not bool((w_scale[:n].abs() > 0).any())
    for attr in ("weight_int8", "weight_packed"):   # OptimizedInt8Conv2d / OptimizedInt4Conv2d
        codes = getattr(mod, attr, None)
        if codes is not None and codes.numel() > 0:
            return not bool((codes != 0).any())
    orig = getattr(mod, "_orig_weight", None)
    if orig is not None:
        # Codes absent or not materialised (int4 keeps an empty `weight_packed` when packing is
        # unavailable): fall back to the fp reference weight these classes retain.
        return not bool((orig != 0).any())
    weight = getattr(mod, "weight", None)
    if weight is None or weight.dim() < 2:
        return False        # norms/embeddings: a zero weight there is not annihilation
    return not bool((weight != 0).any())


def _fp_draw(shape, bound, generator):
    """U(-bound, +bound) of `shape` on CPU fp32 -- the fp weight every mode starts from.

    One code path for all modes: a quantized module quantizes this draw rather than drawing its
    own codes, so the same logical layer holds the same weight at whatever precision the mode uses.
    """
    n = 1
    for d in shape:
        n *= d
    return torch.empty(n, dtype=torch.float32).uniform_(
        -bound, bound, generator=generator).view(*shape)


def _canonical_name(name):
    """One key per LOGICAL layer, so equivalent layers get equal weights across modes.

    The mode conversions wrap and re-expose the same logical layer under different paths:
    FusedResBlock keeps the pre-fusion block under `.original` and re-exposes its output conv as
    `out_conv`, and the int8/int4 conversions replace one of the two with a quantized module,
    leaving the other a live parallel copy. Keying the seed on the raw name would give fp16's
    `...original.out_layers.3` and int8's `...out_conv` different weights, so an fp16-vs-int8
    latent comparison would be measuring that difference rather than quantization.
    """
    return name.replace(".original.", ".").replace("out_conv", "out_layers.3")


def zeroed_modules(model):
    """[(qualified_name, module)] for every weight-bearing module that annihilates its input.

    These are the `zero_module()` sites the stub checkpoint never filled in.
    """
    found, seen = [], set()
    for name, mod in model.named_modules():
        if all(getattr(mod, attr, None) is None
               for attr in ("weight", "qweight", "weight_int8", "weight_packed",
                            "_orig_weight")):
            continue
        if id(mod) in seen:
            # FusedResBlock aliases the same conv as both `original.out_layers.3` and
            # `out_conv`, so named_modules() yields one object twice. Keep the first name.
            continue
        if _effective_weight_is_zero(mod):
            seen.add(id(mod))
            found.append((name, mod))
        else:
            seen.add(id(mod))
    return found


def find_identity_attention_blocks(model):
    """[(name, reason)] for every attention block whose output projection annihilates.

    Decided from the weights rather than by running each block: a zero output projection is a
    sufficient and shape-independent reason, and some routes are shape-specialized.
    """
    dead = []
    for name, block in attention_blocks(model):
        for attr in ("proj", "proj_out"):
            proj = getattr(block, attr, None)
            if proj is not None:
                if _effective_weight_is_zero(proj):
                    dead.append((name, f"{attr} is identically zero"))
                break
    return dead


# ---------------------------------------------------------------- assertions


def assert_attention_observable(model, *, what="this comparison"):
    """Structural check: raise unless every attention block can affect its own output."""
    dead = find_identity_attention_blocks(model)
    if not dead:
        return
    total = len(attention_blocks(model))
    names = ", ".join(name for name, _ in dead[:3])
    more = f" (+{len(dead) - 3} more)" if len(dead) > 3 else ""
    raise NotObservable(
        f"{len(dead)}/{total} attention blocks are identities on their input, so {what} cannot "
        f"fail: {dead[0][1]}, in {names}{more}.\n"
        f"Cause: AttentionBlock.proj_out is a zero_module and this tree's checkpoint is an empty "
        f"stub, so proj_out is never given weights.\n"
        f"Fix: call integration.utils.attention_identity_guard.activate_zeroed_modules(model) "
        f"before comparing, or move the check to the kernel level "
        f"(docs/final_report_2026-07-28/scripts/qattn_correctness.py).")


def assert_unet_output_observable(unet, *, in_channels=4, spatial=32, batch=2,
                                 device="cuda", what="this comparison"):
    """Behavioural check: raise if the UNet predicts identically zero for a random input.

    This is the check that cannot be fooled. `UNetModel.out[-1]` is a zero_module, so with this
    tree's stub checkpoint the epsilon prediction is exactly zero and no latent comparison can
    detect any change anywhere in the network.
    """
    x = torch.randn(batch, in_channels, spatial, spatial, device=device, dtype=torch.float16)
    t = torch.full((batch,), 10, device=device, dtype=torch.long)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
        eps = unet(x, t)
    if bool((eps == 0).all()):
        raise NotObservable(
            f"the UNet predicts identically zero for a random input, so {what} cannot fail: the "
            f"sampled latent depends only on the initial noise and the DDIM schedule, not on "
            f"anything inside the network.\n"
            f"Cause: UNetModel.out[-1] is a zero_module "
            f"(ldm/modules/diffusionmodules/openaimodel.py:745) and this tree's checkpoint is an "
            f"empty stub, so it is never given weights.\n"
            f"Fix: call integration.utils.attention_identity_guard.activate_zeroed_modules("
            f"model) before comparing, or move the check to the kernel level "
            f"(docs/final_report_2026-07-28/scripts/qattn_correctness.py).")


# ---------------------------------------------------------------- activation


def activate_zeroed_modules(model, *, seed=0, verbose=True):
    """Give every annihilating weight real values, in place. Returns the count activated.

    Restores what `zero_module` overwrote: PyTorch's default init for a conv/linear of this
    fan-in, U(-1/sqrt(fan_in), +1/sqrt(fan_in)). Modules with trained weights are left alone, so
    this is a no-op against a real checkpoint.

    Must be called OUTSIDE `torch.inference_mode()` and outside any active `torch.autocast`
    region: autocast caches the fp16 cast of each fp32 weight for the lifetime of the context, so
    a write made inside one is ignored by the rest of that region.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "activate_zeroed_modules() must be called outside torch.inference_mode(); "
            "weights written inside it are not picked up by an enclosing autocast region.")
    activated, skipped = [], []
    for name, mod in zeroed_modules(model):
        # Seed per module from its CANONICAL NAME, not from a single running stream: the set of
        # annihilating modules differs between modes (fp16 has 57, the quantized modes 92), so a
        # shared generator would hand the same layer different weights in different modes and a
        # cross-mode comparison would be measuring that instead of the thing under test.
        generator = torch.Generator(device="cpu").manual_seed(
            (seed + zlib.crc32(_canonical_name(name).encode())) % (2 ** 63))
        qweight = getattr(mod, "qweight", None)
        orig = getattr(mod, "_orig_weight", None)
        if orig is not None or getattr(mod, "weight_int8", None) is not None \
                or getattr(mod, "weight_packed", None) is not None:
            # OptimizedIntXConv2d keeps int codes in NHWC plus SmoothQuant and per-channel
            # scales. Write the fp reference weight it retains, then let it requantize itself
            # so every derived buffer stays consistent.
            fold = getattr(mod, "_fold_weights_with_smooth", None)
            smooth = getattr(mod, "smooth_scale", None)
            if orig is None or fold is None or smooth is None:
                skipped.append(f"{name} (no _orig_weight/_fold_weights_with_smooth)")
                continue
            W = _fp_draw(tuple(orig.shape), 1.0 / math.sqrt(max(orig[0].numel(), 1)), generator)
            with torch.no_grad():
                orig.copy_(W.to(orig.dtype).to(orig.device))
            fold(smooth.detach().reshape(-1).clone())
            activated.append(name)
            continue
        if qweight is not None:
            n, k = mod.out_features, mod.in_features
            q_max = getattr(mod, "Q", 127)
            # Draw the fp weight, then quantize IT -- never draw codes directly. A cross-mode
            # comparison needs each mode to hold the same underlying weight at its own precision;
            # drawing int codes would give fp16 and int4 unrelated weights and the comparison
            # would measure that instead of quantization error. Mirrors QuantLinearWxAx.__init__.
            W = _fp_draw((n, k), 1.0 / math.sqrt(k), generator)
            s = (W.abs().amax(1).clamp_min(1e-8) / q_max)
            codes = torch.round(W / s.unsqueeze(1)).clamp(-q_max, q_max).to(torch.int8)
            if getattr(mod, "bits", 8) == 4:
                from integration.kernels.wxax_linear import _pack4
                # int4 keeps K % 64 == 0, so the logical block is a whole number of packed bytes.
                mod.qweight[:n, :k // 2] = _pack4(codes).to(qweight.device)
            else:
                mod.qweight[:n, :k] = codes.to(qweight.device)
            mod.w_scale[:n] = s.to(mod.w_scale.dtype).to(mod.w_scale.device)
        else:
            weight = mod.weight
            fan_in = weight[0].numel()          # in_features, or in_ch * prod(kernel)
            W = _fp_draw(tuple(weight.shape), 1.0 / math.sqrt(max(fan_in, 1)), generator)
            with torch.no_grad():
                weight.copy_(W.to(weight.dtype).to(weight.device))
        activated.append(name)

    if verbose and activated:
        print(f"  [guard] activated {len(activated)} zero-initialised module(s) so the model "
              f"output is observable: {', '.join(activated[:3])}"
              f"{f' (+{len(activated) - 3} more)' if len(activated) > 3 else ''}")
    if skipped:
        # Never silent: an un-activated annihilating module leaves that part of the graph
        # unobservable, so the caller's comparison is still partly blind there.
        print(f"  [guard] WARNING: could not activate {len(skipped)} annihilating module(s); "
              f"changes behind them remain undetectable: {', '.join(skipped[:3])}"
              f"{f' (+{len(skipped) - 3} more)' if len(skipped) > 3 else ''}")
    return len(activated)


# Previous name, kept so existing callers keep working.
activate_zero_initialised_projections = activate_zeroed_modules


# ---------------------------------------------------------------- call-site helpers

#: Default seed for model CONSTRUCTION. Distinct from the sampling seed on purpose: they control
#: different things and want to be varied independently.
MODEL_SEED = 20260803


def seed_model_construction(seed=MODEL_SEED):
    """Seed the global RNG before building the model. Call this FIRST, before the model exists.

    This tree's checkpoint is an empty stub, so every weight comes from default-initialisation off
    the global RNG -- and torch seeds that generator nondeterministically per process. Unseeded,
    two runs build two DIFFERENT random networks, so nothing about them is comparable: measured
    2026-08-03, a byte-identical rerun of the e2e golden check reported rel_err ~0.4. Seeding here
    makes the network itself reproducible; with it, the same check reports exactly 0.0000.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_for_comparison(model, unet=None, *, what="this comparison", seed=0, verbose=True):
    """Make `model` observable, then assert it is. Call after construction, before comparing.

    Pair with `seed_model_construction()` before the model is built:

        guard.seed_model_construction()
        model, sampler = runner._setup_model(mode)
        guard.prepare_for_comparison(model, what="the A/B below")

    Returns the number of modules activated.
    """
    n = activate_zeroed_modules(model, seed=seed, verbose=verbose)
    if unet is None:
        unet = getattr(getattr(model, "model", None), "diffusion_model", None)
    if unet is not None:
        assert_unet_output_observable(unet, what=what)
    else:
        assert_attention_observable(model, what=what)
    return n
