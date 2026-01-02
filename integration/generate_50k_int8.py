"""
Generate 50k samples using INT8/INT4 MoDiff and calculate FID against real dataset.
"""
import argparse
import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import torchvision.utils as tvu
from tqdm import tqdm

# Disable TF32 globally for consistent benchmarking
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.append(os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# INT8 imports
try:
    from integration.int8_optimized import (
        OptimizedInt8Conv2d,
        convert_model_to_optimized_int8,
        enable_modiff_mode,
        reset_modiff_state,
        set_calibrating,
        get_calibration_config,
        reset_calibration,
    )
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False
    print("Warning: INT8 not available.")

# INT4 imports
try:
    from integration.int4_optimized import (
        OptimizedInt4Conv2d,
        convert_model_to_optimized_int4,
        enable_modiff_mode as enable_modiff_mode_int4,
        reset_modiff_state as reset_modiff_state_int4,
        set_calibrating as set_calibrating_int4,
        get_calibration_config as get_calibration_config_int4,
        reset_calibration as reset_calibration_int4,
    )
    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False
    print("Warning: INT4 not available.")

# FID
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from scipy import linalg
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: pytorch-fid or scipy not installed. FID calculation will be skipped.")

# LSUN Dataset
try:
    from ddim.datasets.lsun import LSUNClass
except ImportError:
    LSUNClass = None


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class LSUNWrapper(Dataset):
    def __init__(self, lsun_dataset):
        self.lsun_dataset = lsun_dataset

    def __len__(self):
        return len(self.lsun_dataset)

    def __getitem__(self, idx):
        img, _ = self.lsun_dataset[idx]
        return img


def get_activations(dataloader, model, device):
    model.eval()
    pred_arr = []
    for batch in tqdm(dataloader, desc="Computing activations"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
    return np.concatenate(pred_arr, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def load_model(config_path: str, ckpt_path: str):
    """Load LDM model from config and checkpoint."""
    print(f"Loading model from {ckpt_path}")
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    
    return model.cuda().eval(), conf


def main():
    parser = argparse.ArgumentParser(description="Generate 50k INT8/INT4 samples and calculate FID")
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--output_dir', type=str, default='results/int8_50k')
    parser.add_argument('--real_dir', type=str, help='Path to real images or pre-calculated .npz stats')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--mode', type=str, default='int8', choices=['int8', 'int4'], help='Quantization mode')
    parser.add_argument('--skip_gen', action='store_true', help='Skip generation, only calculate FID')
    args = parser.parse_args()

    if args.mode == 'int8' and not HAS_INT8:
        print("INT8 mode not available. Exiting.")
        return
    if args.mode == 'int4' and not HAS_INT4:
        print("INT4 mode not available. Exiting.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    if not args.skip_gen:
        # 1. Setup Model
        model, _ = load_model(args.config, args.ckpt)
        
        # Select conversion functions based on mode
        if args.mode == 'int8':
            print(f"Converting UNet to INT8...")
            convert_model_to_optimized_int8(model.model.diffusion_model)
            enable_modiff_mode(model.model.diffusion_model, True)
            reset_calib_fn = reset_calibration
            set_calib_fn = set_calibrating
            get_config_fn = get_calibration_config
            reset_state_fn = reset_modiff_state
        else:  # int4
            print(f"Converting UNet to INT4...")
            convert_model_to_optimized_int4(model.model.diffusion_model)
            enable_modiff_mode_int4(model.model.diffusion_model, True)
            reset_calib_fn = reset_calibration_int4
            set_calib_fn = set_calibrating_int4
            get_config_fn = get_calibration_config_int4
            reset_state_fn = reset_modiff_state_int4
        
        sampler = DDIMSampler(model)
        
        # 2. Calibration (using a few runs)
        print(f"Calibrating {args.mode.upper()}...")
        reset_calib_fn()
        set_calib_fn(model.model.diffusion_model, True)
        with torch.no_grad():
            for _ in range(5):
                reset_state_fn(model.model.diffusion_model)
                sampler.sample(S=5, batch_size=min(args.batch_size, 4), shape=(4, 32, 32), eta=0.0, verbose=False)
        get_config_fn().finalize()
        set_calib_fn(model.model.diffusion_model, False)

        # 3. Generation
        print(f"Generating {args.num_samples} samples in {args.mode.upper()} mode...")
        generated = 0
        pbar = tqdm(total=args.num_samples, desc=f"{args.mode.upper()} Generation")
        
        while generated < args.num_samples:
            batch = min(args.batch_size, args.num_samples - generated)
            reset_state_fn(model.model.diffusion_model)
            
            with torch.no_grad():
                samples, _ = sampler.sample(S=args.steps, batch_size=batch,
                                           shape=(4, 32, 32), eta=0.0, verbose=False)
                
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples.float() + 1.0) / 2.0, 0.0, 1.0)
                
                for i in range(batch):
                    img_id = generated + i
                    tvu.save_image(x_samples[i], os.path.join(sample_dir, f'{img_id:05d}.png'))
            
            generated += batch
            pbar.update(batch)
        pbar.close()
        print(f"Generation complete. Samples saved to {sample_dir}")

    # 4. FID Calculation
    if HAS_FID and args.real_dir:
        print(f"Calculating FID...")
        device = torch.device('cuda')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx]).to(device)
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

        # Generated stats
        gen_dataset = ImageFolderDataset(sample_dir, transform=transform)
        gen_loader = DataLoader(gen_dataset, batch_size=50, shuffle=False, num_workers=4)
        act_gen = get_activations(gen_loader, inception_model, device)
        mu_gen, sigma_gen = np.mean(act_gen, axis=0), np.cov(act_gen, rowvar=False)

        # Real stats
        if os.path.isdir(args.real_dir) and os.path.exists(os.path.join(args.real_dir, 'data.mdb')):
            # LSUN LMDB
            if LSUNClass is None:
                print("Error: Could not load LSUNClass. Check ddim/datasets/lsun.py")
                return
            print(f"Loading LSUN dataset from {args.real_dir}")
            real_dataset = LSUNClass(root=args.real_dir, transform=transform)
            real_dataset = LSUNWrapper(real_dataset)
            print(f"LSUN dataset size: {len(real_dataset)}")
            # Use same number of samples for fair comparison
            if len(real_dataset) > args.num_samples:
                indices = torch.randperm(len(real_dataset))[:args.num_samples].tolist()
                real_dataset = torch.utils.data.Subset(real_dataset, indices)
                print(f"Using subset of {len(real_dataset)} samples")
        else:
            # Image folder
            print(f"Loading images from {args.real_dir}")
            real_dataset = ImageFolderDataset(args.real_dir, transform=transform)
        
        real_loader = DataLoader(real_dataset, batch_size=50, shuffle=False, num_workers=4)
        act_real = get_activations(real_loader, inception_model, device)
        mu_real, sigma_real = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)

        fid_value = calculate_fid(mu_gen, sigma_gen, mu_real, sigma_real)
        print(f"\n{'='*30}\n{args.mode.upper()} FID (50k): {fid_value:.4f}\n{'='*30}")
        
        with open(os.path.join(args.output_dir, 'fid_result.json'), 'w') as f:
            json.dump({'fid': fid_value, 'num_samples': args.num_samples, 'steps': args.steps, 'mode': args.mode}, f)
    elif not HAS_FID:
        print("FID calculation skipped (missing dependencies).")
    elif not args.real_dir:
        print("FID calculation skipped (no real_dir provided).")


if __name__ == '__main__':
    main()
