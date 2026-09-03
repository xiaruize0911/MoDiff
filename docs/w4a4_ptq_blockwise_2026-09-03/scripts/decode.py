import os,sys,torch,numpy as np
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import integration.benchmarks.benchmark_ldm as B
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=8, steps=2, shape=(4,32,32), calibration_path=None, auto_delta_table=False)
m,_=r._setup_model("fp16")
TAGS=["fp16ref","pt_nomig","pt_mig","blk_nomig","blk_mig"]
imgs={}
for t in TAGS:
    lat=torch.load(f"/tmp/w4_lat_{t}.pt",map_location="cuda",weights_only=True); o=[]
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        for i in range(0,lat.shape[0],16):
            o.append(torch.clamp((m.decode_first_stage(lat[i:i+16].float())+1)/2,0,1).float().cpu())
    imgs[t]=torch.cat(o)
ref=imgs["fp16ref"]; a=torch.load("/tmp/w4_lat_fp16ref.pt",weights_only=True)
print(f"\n{'arm':>34} {'image MSE vs fp16':>18} {'PSNR dB':>9} {'latent relL2':>13} {'std':>7}")
for t,lab in [("fp16ref","fp16 (reference)"),("pt_nomig","W4A4 per-tensor, no migration"),
              ("pt_mig","W4A4 per-tensor, + migration"),("blk_nomig","W4A4 blockwise B=64, no mig"),
              ("blk_mig","W4A4 blockwise B=64, + migration")]:
    mse=((imgs[t]-ref)**2).mean().item()
    x=torch.load(f"/tmp/w4_lat_{t}.pt",weights_only=True)
    lr=((x-a).norm()/a.norm()).item() if t!="fp16ref" else 0.0
    ps=10*np.log10(1/mse) if mse>0 else float('inf')
    print(f"{lab:>34} {mse:18.3e} {ps:9.2f} {lr:13.5f} {x.std().item():7.4f}")
import torchvision.utils as vu
vu.save_image(torch.cat([imgs[t][:6] for t in TAGS]),
              "docs/ahat_conv_report_2026-09-02/plots/samples_w4a4_ptq.png",nrow=6)
print("\nwrote plots/samples_w4a4_ptq.png (rows: fp16, pt no-mig, pt +mig, blk no-mig, blk +mig)")
