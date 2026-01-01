import argparse
import os
import sys
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.append(os.getcwd())

# Try to import LSUN dataset class
try:
    from ddim.datasets.lsun import LSUNClass
except ImportError:
    print("Could not import LSUNClass from ddim.datasets.lsun")
    # Fallback or dummy class if needed
    LSUNClass = None

class LSUNWrapper(Dataset):
    def __init__(self, lsun_dataset):
        self.lsun_dataset = lsun_dataset

    def __len__(self):
        return len(self.lsun_dataset)

    def __getitem__(self, idx):
        img, _ = self.lsun_dataset[idx]
        return img

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_activations(dataloader, model, device):
    """Compute Inception activations for a dataloader."""
    pred_arr = []
    
    for batch in tqdm(dataloader, desc="Computing activations"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        
        # If model output is not (N, 2048), reshape it
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
    
    pred_arr = np.concatenate(pred_arr, axis=0)
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True, help='Path to generated images folder')
    parser.add_argument('--dataset_path', type=str, default='data/lsun/churches_train', help='Path to LSUN LMDB dataset')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_real_samples', type=int, default=10000, help='Number of real samples to use')
    parser.add_argument('--num_gen_samples', type=int, default=None, help='Number of generated samples to use')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Inception Model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # Transform for Inception (Resize to 299x299, Normalize)
    # Note: pytorch_fid handles normalization internally if inputs are 0-1 tensors? 
    # Actually pytorch_fid expects 0-1 float tensors or 0-255 uint8 tensors.
    # We will provide 0-1 float tensors.
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # 2. Setup Generated Data Loader
    print(f"Loading generated images from {args.generated}...")
    gen_dataset = ImageFolderDataset(args.generated, transform=transform)
    if args.num_gen_samples is not None and args.num_gen_samples < len(gen_dataset):
        indices = torch.randperm(len(gen_dataset))[:args.num_gen_samples]
        gen_dataset = torch.utils.data.Subset(gen_dataset, indices)
    print(f"Generated dataset size: {len(gen_dataset)}")
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Setup Real Data Loader
    print(f"Loading real LSUN images from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist.")
        print("Please provide the correct path to the LSUN LMDB folder.")
        return

    try:
        real_dataset = LSUNClass(root=args.dataset_path, transform=transform)
        real_dataset = LSUNWrapper(real_dataset)
        print(f"Real dataset size (full): {len(real_dataset)}")
        # Limit number of real samples to save time/memory if needed
        if args.num_real_samples < len(real_dataset):
            indices = torch.randperm(len(real_dataset))[:args.num_real_samples]
            real_dataset = torch.utils.data.Subset(real_dataset, indices)
        
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Failed to load LSUN dataset: {e}")
        return

    # 4. Compute Statistics
    print("Computing statistics for generated images...")
    act_gen = get_activations(gen_loader, model, device)
    mu_gen = np.mean(act_gen, axis=0)
    sigma_gen = np.cov(act_gen, rowvar=False)

    print("Computing statistics for real images...")
    act_real = get_activations(real_loader, model, device)
    mu_real = np.mean(act_real, axis=0)
    sigma_real = np.cov(act_real, rowvar=False)

    # 5. Calculate FID
    print("Calculating FID...")
    fid_value = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    
    print(f"\n{'='*40}")
    print(f"FID Score: {fid_value:.4f}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()
