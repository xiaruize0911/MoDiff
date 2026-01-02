"""
Calculate FID score with LMDB dataset support.
"""
import argparse
import os
import lmdb
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pytorch_fid.inception import InceptionV3


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, max_samples=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
        if max_samples:
            self.length = min(self.length, max_samples)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            for i in range(idx):
                cursor.next()
            _, value = cursor.item()
        
        img = Image.open(BytesIO(value)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_activations(dataloader, model, device, max_samples=None):
    model.eval()
    pred_arr = []
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing activations"):
            if max_samples and total >= max_samples:
                break
            
            batch = batch.to(device)
            pred = model(batch)[0]
            
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
            total += len(batch)
    
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


def main():
    parser = argparse.ArgumentParser(description="Calculate FID with LMDB support")
    parser.add_argument('gen_dir', type=str, help='Path to generated images')
    parser.add_argument('real_dir', type=str, help='Path to real images or LMDB')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, help='Output file for FID result')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load Inception model
    print("Loading Inception model...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # Load generated images
    print(f"\nLoading generated images from {args.gen_dir}")
    gen_dataset = ImageFolderDataset(args.gen_dir, transform=transform)
    print(f"Found {len(gen_dataset)} generated images")
    
    if len(gen_dataset) > args.num_samples:
        indices = list(range(args.num_samples))
        gen_dataset = torch.utils.data.Subset(gen_dataset, indices)
        print(f"Using first {args.num_samples} images")
    
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # Load real images (LMDB or folder)
    print(f"\nLoading real images from {args.real_dir}")
    if os.path.exists(os.path.join(args.real_dir, 'data.mdb')):
        print("Detected LMDB database")
        real_dataset = LMDBDataset(args.real_dir, transform=transform, max_samples=args.num_samples)
    else:
        real_dataset = ImageFolderDataset(args.real_dir, transform=transform)
        if len(real_dataset) > args.num_samples:
            indices = torch.randperm(len(real_dataset))[:args.num_samples].tolist()
            real_dataset = torch.utils.data.Subset(real_dataset, indices)
    
    print(f"Using {len(real_dataset)} real images")
    
    # LMDB doesn't work well with multiple workers
    num_workers = 0 if os.path.exists(os.path.join(args.real_dir, 'data.mdb')) else 2
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Compute activations
    print("\n" + "="*60)
    print("Computing activations for generated images...")
    act_gen = get_activations(gen_loader, inception_model, device, max_samples=args.num_samples)
    mu_gen = np.mean(act_gen, axis=0)
    sigma_gen = np.cov(act_gen, rowvar=False)
    print(f"Generated: {act_gen.shape[0]} samples")
    
    print("\nComputing activations for real images...")
    act_real = get_activations(real_loader, inception_model, device, max_samples=args.num_samples)
    mu_real = np.mean(act_real, axis=0)
    sigma_real = np.cov(act_real, rowvar=False)
    print(f"Real: {act_real.shape[0]} samples")
    
    # Calculate FID
    print("\nCalculating FID...")
    fid_value = calculate_fid(mu_gen, sigma_gen, mu_real, sigma_real)
    
    print("="*60)
    print(f"FID Score: {fid_value:.4f}")
    print(f"Generated samples: {act_gen.shape[0]}")
    print(f"Real samples: {act_real.shape[0]}")
    print("="*60)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"{fid_value:.4f}\n")
        print(f"\nFID saved to {args.output}")
    
    return fid_value


if __name__ == '__main__':
    main()
