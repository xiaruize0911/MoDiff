import os,sys,time,statistics; os.chdir("/workspace/MoDiff"); sys.path.insert(0,"/workspace/MoDiff"); sys.path.insert(0,"/workspace/MoDiff/src/taming-transformers")
import torch
def run(backend, measure_latent=False):
    os.environ["MODIFF_SDPA_BACKEND"]=backend
    os.environ["MODIFF_QUANT_LINEAR"]="0"; os.environ["MODIFF_QUANT_ATTN"]="0"
    # fresh import so _SDPA_CTX picks up the env
    for m in list(sys.modules):
        if "token_major_attention" in m: del sys.modules[m]
    import integration.benchmarks.benchmark_ldm as B
    r=B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml","models/ldm/lsun_churches256/model.ckpt",output_dir="integration/results/ph1",batch_size=128,steps=2,shape=(4,32,32),calibration_path=None,linear_backend="fp16")
    model,sampler=r._setup_model("fp16"); dm=model.model.diffusion_model
    g=torch.Generator(device="cuda").manual_seed(0)
    x=torch.randn(128,4,32,32,device="cuda",generator=g); t=torch.randint(0,1000,(128,),device="cuda",generator=torch.Generator(device="cuda").manual_seed(1))
    def step(): 
        with torch.inference_mode(), torch.amp.autocast('cuda',dtype=torch.float16): return dm(x,t)
    out=None
    if measure_latent:
        with torch.inference_mode(), torch.amp.autocast('cuda',dtype=torch.float16): out=step().float().clone()
    for _ in range(30): step()
    torch.cuda.synchronize(); ts=[]
    for _ in range(3):
        torch.cuda.synchronize(); s=time.time()
        for _ in range(40): step()
        torch.cuda.synchronize(); ts.append((time.time()-s)/40*1000)
    ts.sort(); return ts[1], out
tm,out_m=run("math",measure_latent=True)
tf,out_f=run("flash",measure_latent=True)
rel=(out_f-out_m).norm().item()/(out_m.norm().item()+1e-9)
print(f"\nfp16 MATH   {tm:.2f} ms/step")
print(f"fp16 FLASH  {tf:.2f} ms/step   ({tm/tf:.2f}x vs MATH)")
print(f"latent rel-err flash vs math: {rel:.2e}  {'OK (<1e-3)' if rel<1e-3 else 'CHECK'}")
