"""
FID Evaluation Module

Computes FID score for generated samples against reference dataset.

Usage:
    python -m demo_int8_modiff.fid_evaluation \
        --gen_dir samples/cifar10_int8 \
        --ref_stats calibration/cifar10_fid_stats.npz \
        --batch_size 128

Following the same methodology as the original MoDiff evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_inception_model(device: str = 'cuda') -> nn.Module:
    """
    Load InceptionV3 model for FID computation.
    Uses pool3 features (2048-dimensional).
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    except (ImportError, TypeError):
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=True)
    
    # Remove final layers, keep up to pool3
    model.fc = nn.Identity()
    model.dropout = nn.Identity()
    
    model.eval()
    model.to(device)
    
    return model


def compute_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Inception features for a dataset.
    """
    features = []
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing features"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            
            # Resize to 299x299 for InceptionV3
            if images.shape[-1] != 299:
                images = F.interpolate(
                    images, size=(299, 299),
                    mode='bilinear', align_corners=False
                )
            
            # Normalize to [-1, 1] if needed
            if images.min() >= 0 and images.max() <= 1:
                images = 2 * images - 1
            
            # Forward pass
            feats = model(images)
            features.append(feats.cpu().numpy())
            
            total += images.shape[0]
            if max_samples is not None and total >= max_samples:
                break
    
    features = np.concatenate(features, axis=0)
    if max_samples is not None:
        features = features[:max_samples]
    
    return features


def compute_fid_from_features(
    features_real: np.ndarray,
    features_fake: np.ndarray,
) -> float:
    """
    Compute FID from pre-computed features.
    
    FID = ||mu_real - mu_fake||^2 + Tr(Sigma_real + Sigma_fake - 2*sqrt(Sigma_real @ Sigma_fake))
    """
    from scipy import linalg
    
    # Compute statistics
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)
    
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_fake
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    # Handle numerical errors
    if not np.isfinite(covmean).all():
        logger.warning("FID calculation produced non-finite values, adding epsilon")
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logger.warning(f"Imaginary component {m} in FID computation")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean
    
    return float(fid)


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder."""
    
    def __init__(
        self,
        folder: str,
        transform: Optional[transforms.Compose] = None,
        max_images: Optional[int] = None,
    ):
        self.folder = Path(folder)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Find all images
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        self.images = sorted([
            p for p in self.folder.iterdir()
            if p.suffix.lower() in extensions
        ])
        
        if max_images is not None:
            self.images = self.images[:max_images]
        
        logger.info(f"Found {len(self.images)} images in {folder}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


class NPZDataset(Dataset):
    """Dataset for loading images from NPZ files."""
    
    def __init__(
        self,
        folder: str,
        transform: Optional[transforms.Compose] = None,
        max_images: Optional[int] = None,
    ):
        self.folder = Path(folder)
        self.transform = transform
        
        # Find all npz files
        self.npz_files = sorted(self.folder.glob("*.npz"))
        
        # Load all samples
        all_samples = []
        for npz_file in self.npz_files:
            data = np.load(npz_file)
            if 'samples' in data:
                all_samples.append(data['samples'])
            elif 'arr_0' in data:
                all_samples.append(data['arr_0'])
        
        if all_samples:
            self.samples = np.concatenate(all_samples, axis=0)
        else:
            self.samples = np.array([])
        
        if max_images is not None:
            self.samples = self.samples[:max_images]
        
        logger.info(f"Loaded {len(self.samples)} samples from {folder}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        
        # Convert to tensor
        if sample.dtype == np.uint8:
            sample = sample.astype(np.float32) / 255.0
        
        # Ensure CHW format
        if sample.ndim == 3 and sample.shape[-1] in [1, 3]:
            sample = np.transpose(sample, (2, 0, 1))
        
        sample = torch.from_numpy(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def compute_reference_stats(
    ref_dataset: str,
    output_path: str,
    batch_size: int = 128,
    device: str = 'cuda',
    max_samples: int = 50000,
):
    """
    Compute and save reference dataset statistics.
    """
    logger.info(f"Computing reference statistics for {ref_dataset}")
    
    # Load Inception model
    inception = get_inception_model(device)
    
    # Load dataset
    if ref_dataset.lower() == 'cifar10':
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        dataset = ImageFolderDataset(ref_dataset, max_images=max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Compute features
    features = compute_features(inception, dataloader, device, max_samples)
    
    # Compute statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, mu=mu, sigma=sigma, features=features)
    
    logger.info(f"Saved reference statistics to {output_path}")


def compute_fid(
    gen_path: str,
    ref_stats_path: Optional[str] = None,
    ref_dataset: str = 'cifar10',
    batch_size: int = 128,
    device: str = 'cuda',
    max_samples: int = 50000,
) -> float:
    """
    Compute FID score.
    
    Args:
        gen_path: Path to generated samples (folder or npz)
        ref_stats_path: Path to reference statistics (optional)
        ref_dataset: Reference dataset name or path
        batch_size: Batch size for feature computation
        device: Device to use
        max_samples: Maximum number of samples to use
    
    Returns:
        FID score
    """
    # Load Inception model
    inception = get_inception_model(device)
    
    # Load generated samples
    gen_path = Path(gen_path)
    if gen_path.is_dir():
        if any(gen_path.glob("*.npz")):
            gen_dataset = NPZDataset(gen_path, max_images=max_samples)
        else:
            gen_dataset = ImageFolderDataset(gen_path, max_images=max_samples)
    else:
        # Single npz file
        data = np.load(gen_path)
        if 'samples' in data:
            samples = data['samples']
        else:
            samples = data['arr_0']
        
        class ArrayDataset(Dataset):
            def __init__(self, arr):
                self.arr = arr
            def __len__(self):
                return len(self.arr)
            def __getitem__(self, idx):
                s = self.arr[idx]
                if s.dtype == np.uint8:
                    s = s.astype(np.float32) / 255.0
                if s.ndim == 3 and s.shape[-1] in [1, 3]:
                    s = np.transpose(s, (2, 0, 1))
                return torch.from_numpy(s)
        
        gen_dataset = ArrayDataset(samples[:max_samples])
    
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Compute generated features
    logger.info("Computing features for generated samples...")
    gen_features = compute_features(inception, gen_loader, device, max_samples)
    
    # Load or compute reference features
    if ref_stats_path is not None and os.path.exists(ref_stats_path):
        logger.info(f"Loading reference statistics from {ref_stats_path}")
        ref_data = np.load(ref_stats_path)
        ref_features = ref_data['features']
    else:
        logger.info(f"Computing reference features for {ref_dataset}")
        
        if ref_dataset.lower() == 'cifar10':
            from torchvision.datasets import CIFAR10
            ref_ds = CIFAR10(
                root='data',
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
        else:
            ref_ds = ImageFolderDataset(ref_dataset, max_images=max_samples)
        
        ref_loader = DataLoader(
            ref_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        ref_features = compute_features(inception, ref_loader, device, max_samples)
    
    # Compute FID
    logger.info("Computing FID score...")
    fid = compute_fid_from_features(ref_features, gen_features)
    
    return fid


def main():
    parser = argparse.ArgumentParser(description="FID Evaluation")
    
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Path to generated samples")
    parser.add_argument("--ref_stats", type=str, default=None,
                        help="Path to reference statistics")
    parser.add_argument("--ref_dataset", type=str, default="cifar10",
                        help="Reference dataset (cifar10 or path)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=50000)
    
    parser.add_argument("--compute_ref_stats", action="store_true",
                        help="Compute and save reference statistics")
    parser.add_argument("--output_stats", type=str, 
                        default="calibration/cifar10_fid_stats.npz",
                        help="Output path for reference statistics")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    if args.compute_ref_stats:
        compute_reference_stats(
            args.ref_dataset,
            args.output_stats,
            args.batch_size,
            device,
            args.max_samples,
        )
    else:
        fid = compute_fid(
            args.gen_dir,
            args.ref_stats,
            args.ref_dataset,
            args.batch_size,
            device,
            args.max_samples,
        )
        
        logger.info(f"FID Score: {fid:.2f}")
        print(f"\n{'='*50}")
        print(f"FID Score: {fid:.4f}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
