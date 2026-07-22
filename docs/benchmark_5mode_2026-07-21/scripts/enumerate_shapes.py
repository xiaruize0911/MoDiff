"""Ground-truth enumeration of EVERY kernel shape the churches UNet dispatches per DDIM step,
with its per-step call count, for all 5 modes.

Method: build the real model via BenchmarkRunner._setup_model(mode) (so the actual, latest fused
int8/int4 conv/linear/attention kernels are wired), register forward hooks on every conv / linear /
attention module, run N real DDIM steps, and count how many times each (family, shape, kernel-class)
fires. count_per_step = total_fires / N (asserted integer). This is the authoritative "all shapes each
kernel runs on + how many times it runs in one step" table that drives every kernel benchmark.

Writes data/kernel_shapes.csv (one row per mode x family x distinct shape).
"""
import os, sys, csv, json
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn as nn
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
NSTEPS = 4                       # count over N real steps, divide -> per-step count
HERE = "docs/benchmark_5mode_2026-07-21"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

CONV_CLASSES = {"OptimizedInt8Conv2d", "OptimizedInt4Conv2d"}
LIN_CLASSES = {"QuantLinearWxAx", "OptimizedInt8Linear", "OptimizedInt4Linear"}
ATTN_CLASSES = {"QuantizedStandardAttentionBlock", "TokenMajorAttentionBlock", "AttentionBlock"}


def classify_lin(name):
    n = name.lower()
    if n.endswith(".qkv") or n.endswith("qkv"): return "qkv"
    if n.endswith(".proj") or n.endswith("proj") or "proj_out" in n: return "proj"
    return "other"


def run(mode_label, mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                          batch_size=BATCH, steps=NSTEPS, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    dm = model.model.diffusion_model

    # Per-UNet-forward (= per DDIM step) counting. The UNetModel.forward pre-hook opens a fresh
    # per-step dict; each kernel hook writes into the current step. Modiff's first step (cache
    # init) differs from steady state, so we report the LAST step's counts (the steady-state cost
    # paid on all but the first of the sampler's steps) and assert the last two steps agree.
    steps = []                # list of per-step dicts: key -> entry
    cur = {}
    handles = []

    def new_step(m, inp):
        nonlocal cur
        cur = {}
        steps.append(cur)

    def _conv_ks(m):
        ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        st = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
        pd = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
        return ks[0], st[0], pd[0]

    def record_conv(m, name, cls, Hh, Ww):
        K, st, pd = _conv_ks(m)
        Cin = getattr(m, "in_channels")
        key = ("conv", cls, Cin, Hh, Ww, m.out_channels, K, st, pd)
        e = cur.setdefault(key, dict(family="conv", kernel_class=cls, Cin=Cin, H=Hh, W=Ww,
                                     Cout=m.out_channels, K=K, stride=st, pad=pd, count=0, names=set()))
        e["count"] += 1; e["names"].add(name)

    def conv_hook(name, cls):        # plain fp16 nn.Conv2d (called via __call__)
        def h(m, inp):
            x = inp[0]
            record_conv(m, name, cls, x.shape[2], x.shape[3])
        return h

    def wrap_conv_methods(m, name, cls):
        """Optimized int8/int4 convs are invoked by the fused resblock through methods that
        BYPASS __call__ (forward_from_int8/int4[_dual], forward_modiff_fused_silu_residual,
        forward_gn_fused_modiff) as well as forward(). Each such method runs exactly one conv
        kernel. Wrap them all (per instance) with a reentrancy guard so we count 1 per kernel."""
        specs = {   # method -> callable(args,kwargs) -> (H, W)
            "forward":                          lambda a, k: (a[0].shape[2], a[0].shape[3]),
            "forward_from_int8":                lambda a, k: (a[0].shape[2], a[0].shape[3]),
            "forward_from_int8_dual":           lambda a, k: (a[0].shape[2], a[0].shape[3]),
            "forward_from_int4":                lambda a, k: (a[1], a[2]),                       # (packed, h_in, w_in, ...)
            "forward_from_int4_dual":           lambda a, k: (a[1], a[2]),
            "forward_modiff_fused_silu_residual": lambda a, k: (a[0].shape[2], a[0].shape[3]),
            "forward_gn_fused_modiff":          lambda a, k: (a[0].shape[2], a[0].shape[3]),
        }
        for mname, hw in specs.items():
            if not hasattr(m, mname):
                continue
            orig = getattr(m, mname)
            def make(orig=orig, hw=hw):
                def wrapper(*a, **k):
                    if not getattr(m, "_enum_active", False):
                        m._enum_active = True
                        try:
                            Hh, Ww = hw(a, k)
                            record_conv(m, name, cls, Hh, Ww)
                        except Exception as ex:
                            print(f"  (enum warn {name}.{orig.__name__}: {ex})")
                        try:
                            return orig(*a, **k)
                        finally:
                            m._enum_active = False
                    return orig(*a, **k)
                return wrapper
            setattr(m, mname, make())

    def lin_hook(name, cls, role):
        def h(m, inp):
            x = inp[0]
            K = x.shape[-1]; M = 1
            for s in x.shape[:-1]: M *= s
            N = getattr(m, "out_features", None) or getattr(m, "N", None)
            key = ("linear", cls, role, K, N, M)
            e = cur.setdefault(key, dict(family="linear", kernel_class=cls, role=role, K=K, N=N, M=M,
                                         count=0, names=set()))
            e["count"] += 1; e["names"].add(name)
        return h

    def wrap_lin_from_int8(m, name, cls, role):
        """When the GN->qkv-quantize fusion is on (default), the qkv linear is called via
        forward_from_int8(x_i8[M,K]) which BYPASSES __call__, so the forward_pre_hook misses it.
        Wrap the method too (one GEMM per call, no reentrancy into forward) so the qkv GEMM is
        still counted -> the inventory stays fusion-independent (89/79/21)."""
        if not hasattr(m, "forward_from_int8"):
            return
        orig = m.forward_from_int8

        def wrapper(x_i8, residual=None, *a, **k):
            K = m.in_features; N = m.out_features; M = int(x_i8.shape[0])
            key = ("linear", cls, role, K, N, M)
            e = cur.setdefault(key, dict(family="linear", kernel_class=cls, role=role, K=K, N=N, M=M,
                                         count=0, names=set()))
            e["count"] += 1; e["names"].add(name)
            return orig(x_i8, residual, *a, **k)
        m.forward_from_int8 = wrapper

    def attn_hook(name, cls):
        def h(m, inp):
            x = inp[0]
            b, C, Hh, Ww = x.shape; T = Hh * Ww
            nh = getattr(m, "num_heads"); hd = getattr(m, "head_dim", C // nh)
            elig = int(hd <= 48 and T % 64 == 0)
            key = ("attn", cls, C, nh, hd, T)
            e = cur.setdefault(key, dict(family="attn", kernel_class=cls, C=C, nh=nh, hd=hd, T=T,
                                         Hspatial=Hh, flash_eligible=elig, count=0, names=set()))
            e["count"] += 1; e["names"].add(name)
        return h

    handles.append(dm.register_forward_pre_hook(new_step))
    for name, m in dm.named_modules():
        cls = type(m).__name__
        if cls in ATTN_CLASSES:
            handles.append(m.register_forward_pre_hook(attn_hook(name, cls)))
        elif cls in CONV_CLASSES:
            wrap_conv_methods(m, name, cls)          # method-level (fused path bypasses __call__)
        elif isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_pre_hook(conv_hook(name, cls)))
        elif cls in LIN_CLASSES or isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(lin_hook(name, cls, classify_lin(name))))
            wrap_lin_from_int8(m, name, cls, classify_lin(name))  # fused qkv bypasses __call__

    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=NSTEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    for hd_ in handles: hd_.remove()

    # steady-state = last step; verify last two steps identical in counts
    def counts_only(d): return {k: v["count"] for k, v in d.items()}
    assert len(steps) == NSTEPS, f"{mode_label}: saw {len(steps)} UNet forwards, expected {NSTEPS}"
    if counts_only(steps[-1]) != counts_only(steps[-2]):
        print(f"  WARNING {mode_label}: last two steps differ (using last/steady state)")
    canon = steps[-1]
    first = steps[0]
    fc = counts_only(first)
    rows = []
    for key, e in canon.items():
        cnt = e.pop("count"); names = e.pop("names")
        e["mode"] = mode_label
        e["count_per_step"] = cnt
        e["count_first_step"] = fc.get(key, 0)
        e["n_layers"] = len(names)
        rows.append(e)
    del model, sampler; torch.cuda.empty_cache()
    return rows


allrows = []
for (label, mode) in VERS:
    print(f"\n===== enumerating {label} =====")
    rr = run(label, mode)
    conv = [r for r in rr if r["family"] == "conv"]
    lin = [r for r in rr if r["family"] == "linear"]
    attn = [r for r in rr if r["family"] == "attn"]
    print(f"  conv: {len(conv)} distinct shapes, {sum(r['count_per_step'] for r in conv)} calls/step")
    print(f"  linear: {len(lin)} distinct shapes, {sum(r['count_per_step'] for r in lin)} calls/step")
    print(f"  attn: {len(attn)} distinct shapes, {sum(r['count_per_step'] for r in attn)} calls/step")
    allrows += rr

cols = ["mode", "family", "kernel_class", "role", "Cin", "H", "W", "Cout", "K", "stride", "pad",
        "C", "nh", "hd", "T", "Hspatial", "flash_eligible", "M", "N",
        "count_per_step", "count_first_step", "n_layers"]
with open(f"{HERE}/data/kernel_shapes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in allrows: w.writerow(r)
print(f"\nWROTE {HERE}/data/kernel_shapes.csv  ({len(allrows)} rows)")
