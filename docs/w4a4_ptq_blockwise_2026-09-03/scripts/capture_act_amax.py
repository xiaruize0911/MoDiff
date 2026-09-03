"""Per-input-channel activation amax at every 3x3 conv, from a real fp16 sampling run.
Needed for SmoothQuant-style migration lambda_j = max|X_j|^a / max|W_j|^(1-a); the repo has a
smooth_scale buffer but it is identity, and the calibration file is one per-tensor float per layer.
fp16 mode has no fused quantized conv path, so ordinary forward hooks reach the true conv inputs."""
import os,sys,torch
ROOT="/workspace/MoDiff"; sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]; os.chdir(ROOT)
import integration.benchmarks.benchmark_ldm as B
BATCH,STEPS=4,10
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=BATCH, steps=STEPS, shape=(4,32,32), calibration_path=None, auto_delta_table=False)
m,s=r._setup_model("fp16"); unet=m.model.diffusion_model
amax={}; hooks=[]
def mk(name):
    def h(mod,inp,out):
        x=inp[0].detach().float()
        a=x.abs().amax(dim=(0,2,3))
        amax[name]=a if name not in amax else torch.maximum(amax[name],a)
    return h
n=0
for name,mod in unet.named_modules():
    if isinstance(mod,torch.nn.Conv2d) and mod.kernel_size==(3,3):
        hooks.append(mod.register_forward_hook(mk(name))); n+=1
print(f"hooked {n} 3x3 convs")
torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
    s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
for h in hooks: h.remove()
print(f"captured {len(amax)} layers")
torch.save({k:v.cpu() for k,v in amax.items()}, "/tmp/act_amax.pt")
