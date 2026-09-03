import os,sys,torch,numpy as np
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import integration.benchmarks.benchmark_ldm as B
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=8, steps=2, shape=(4,32,32), calibration_path=None, auto_delta_table=False)
m,_=r._setup_model("fp16")
TAGS=["ref","refb","b10sr1","b8sr1","b8sr0","b6sr1","b6sr0","b4sr1","b4sr0"]
imgs={}
for t in TAGS:
    lat=torch.load(f"/tmp/oh_lat_int8_{t}.pt",map_location="cuda",weights_only=True)
    o=[]
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        for i in range(0,lat.shape[0],16):
            o.append(torch.clamp((m.decode_first_stage(lat[i:i+16].float())+1)/2,0,1).float().cpu())
    imgs[t]=torch.cat(o); print(f"decoded {t}",flush=True)
ref=imgs["ref"]
print(f"\n{'arm':>8} {'image MSE vs fp16 o_hat':>24} {'/ floor':>8} {'PSNR dB':>9}  resolvable?")
fl=((imgs["refb"]-ref)**2).mean().item()
for t in TAGS[1:]:
    mse=((imgs[t]-ref)**2).mean().item()
    print(f"{t:>8} {mse:24.3e} {mse/fl:8.2f}x {10*np.log10(1/mse):9.2f}  "
          f"{'yes' if mse>3*fl else 'NO'}{'   <- FLOOR' if t=='refb' else ''}")
import torchvision.utils as vu
g=torch.cat([imgs[t][:6] for t in ("ref","b10sr1","b8sr1","b8sr0","b6sr1","b6sr0")])
vu.save_image(g,"docs/ahat_conv_report_2026-09-02/plots/samples_ohat.png",nrow=6)
print("\nwrote plots/samples_ohat.png (rows: fp16, 10b+SR, 8b+SR, 8b no-SR, 6b+SR, 6b no-SR)")
