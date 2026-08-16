"""Cross-arm output comparison: does MODIFF_GN_STATS_FAST actually change the kernel's numbers?

C10 switches a block size, not an entry point, so a kernel-name counter cannot see it -- the usual
non-vacuity check is blind here. This runs the fused kernel on the SAME captured inputs under whichever
policy the environment selects and dumps its outputs, so two invocations can be differenced. If the two
policies produce byte-identical output, the flag is inert and any A/B built on it is vacuous.
"""
import os, sys, torch
os.chdir('/workspace/MoDiff'); sys.path.insert(0,'.')
os.environ.setdefault("MODIFF_QUANT_LINEAR","1"); os.environ.setdefault("MODIFF_QUANT_ATTN","1")
import modiff_cutlass as M
cl=lambda t: t.contiguous(memory_format=torch.channels_last) if t.dim()==4 else t.contiguous()
b=torch.load(sys.argv[1], map_location='cuda', weights_only=False)
out=[]
for c in b['cases']:
    x=cl(c['x'].cuda()); w=c['w'].cuda(); bb=c['b'].cuda(); ms=c['ms'].cuda(); sh=c['sh'].cuda()
    scale=c['scale'].cuda(); smooth=c['smooth'].cuda(); ng=c['ng']; eps=c['eps']
    dyn=[d.cuda() if torch.is_tensor(d) else d for d in c['dyn_t']]
    with torch.inference_mode():
        a=cl(c['a_hat'].cuda())
        q=M.group_norm_silu_delta_quantize_nhwc(x,w,bb,a,ng,eps,True,scale,smooth,ms,sh,*dyn)
        torch.cuda.synchronize()
    out.append({"q":q.cpu().clone(),"a":a.cpu().clone()})
torch.save(out, sys.argv[2])
print(f"dumped {len(out)} cases -> {sys.argv[2]}")
