"""
Extract images from LSUN LMDB dataset to a folder.
"""
import argparse
import os
import lmdb
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def extract_lmdb(lmdb_path, output_dir, max_images=50000):
    os.makedirs(output_dir, exist_ok=True)
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        
        count = 0
        for key, value in tqdm(cursor, desc="Extracting images", total=max_images):
            if count >= max_images:
                break
            
            img = Image.open(BytesIO(value))
            img = img.convert('RGB')
            
            # Save image
            img_path = os.path.join(output_dir, f'{count:05d}.png')
            img.save(img_path)
            
            count += 1
    
    env.close()
    print(f"Extracted {count} images to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract images from LSUN LMDB")
    parser.add_argument('lmdb_path', type=str, help='Path to LSUN LMDB directory')
    parser.add_argument('output_dir', type=str, help='Output directory for images')
    parser.add_argument('--max_images', type=int, default=50000, help='Maximum images to extract')
    args = parser.parse_args()
    
    extract_lmdb(args.lmdb_path, args.output_dir, args.max_images)
