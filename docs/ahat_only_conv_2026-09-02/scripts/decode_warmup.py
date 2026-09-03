"""Decode the saved latents and compare in image space, against the calibrated 1.705e-03
run-to-run floor from REPORT.md section 5."""
import os,sys,torch,numpy as np
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import integration.benchmarks.benchmark_ldm as B
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=8, steps=2, shape=(4,32,32), calibration_path=None, auto_delta_table=False)
m,_=r._setup_model("fp16")
imgs={}
for tag in ("wu5","wu5b","wu4","wu3","wu2","wu1"):
    lat=torch.load(f"/tmp/lat_int4_{tag}.pt",map_location="cuda",weights_only=True)
    outs=[]
    with torch.inference_mode():
        for i in range(0,lat.shape[0],16):
            with torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
                x=m.decode_first_stage(lat[i:i+16].float())
            outs.append(torch.clamp((x+1.0)/2.0,0,1).float().cpu())
    imgs[tag]=torch.cat(outs)
    print(f"decoded {tag}: {tuple(imgs[tag].shape)}",flush=True)
ref=imgs["wu5"]; FLOOR=1.705e-3
print(f"\n{'arm':>6} {'image MSE vs wu=5':>18} {'/ floor':>8} {'PSNR dB':>9}  resolvable?")
for tag in ("wu5b","wu4","wu3","wu2","wu1"):
    mse=((imgs[tag]-ref)**2).mean().item()
    print(f"{tag:>6} {mse:18.3e} {mse/FLOOR:8.2f}x {10*np.log10(1.0/mse):9.2f}  {'yes' if mse>3*FLOOR else 'NO'}")
import torchvision.utils as vu
grid=torch.cat([imgs[t][:6] for t in ("wu5","wu4","wu3","wu2","wu1")])
os.makedirs("docs/ahat_only_conv_2026-09-02/samples",exist_ok=True)
vu.save_image(grid,"docs/ahat_conv_report_2026-09-02/plots/samples_warmup.png",nrow=6)
print("\nwrote samples/warmup_sweep.png (rows: wu=5,4,3,2,1)")
