import os,sys,torch,numpy as np
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import integration.benchmarks.benchmark_ldm as B
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=8, steps=2, shape=(4,32,32), calibration_path=None, auto_delta_table=False)
m,_=r._setup_model("fp16")
TAGS=["fp16ref","split_lin","split_conv","blk_mig"]
imgs={}
for t in TAGS:
    lat=torch.load(f"/tmp/w4_lat_{t}.pt",map_location="cuda",weights_only=True); o=[]
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        for i in range(0,lat.shape[0],16):
            o.append(torch.clamp((m.decode_first_stage(lat[i:i+16].float())+1)/2,0,1).float().cpu())
    imgs[t]=torch.cat(o)
ref=imgs["fp16ref"]; a=torch.load("/tmp/w4_lat_fp16ref.pt",weights_only=True)
print(f"\n{'arm':>36} {'image MSE vs fp16':>18} {'PSNR dB':>9} {'latent relL2':>13}")
for t,lab in [("split_lin","W4A4 LINEAR layers only"),("split_conv","W4A4 CONV layers only"),
              ("blk_mig","W4A4 conv blockwise + migration")]:
    mse=((imgs[t]-ref)**2).mean().item()
    x=torch.load(f"/tmp/w4_lat_{t}.pt",weights_only=True)
    print(f"{lab:>36} {mse:18.3e} {10*np.log10(1/mse):9.2f} {((x-a).norm()/a.norm()).item():13.5f}")
import torchvision.utils as vu
vu.save_image(torch.cat([imgs[t][:6] for t in ("fp16ref","split_lin","split_conv")]),
              "docs/w4a4_ptq_blockwise_2026-09-03/samples_split.png",nrow=6)
print("\nwrote samples_split.png (rows: fp16, linear-only W4A4, conv-only W4A4)")
