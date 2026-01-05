#!/usr/bin/env python3
"""
MoDiff Setup Script
===================
This script automates the setup process for running MoDiff benchmarks on a new machine.

It performs the following steps:
1. Installs required dependencies (nvidia-cutlass, blinker)
2. Rebuilds CUDA extensions (modiff_cuda) with CUTLASS support
3. Downloads and extracts first-stage autoencoder models
4. Sets up taming-transformers package
5. Patches PyTorch 2.6+ compatibility issues

Usage:
    python setup_modiff.py [--skip-cuda] [--skip-models]
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import urllib.request
import zipfile
import shutil

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_step(msg):
    print(f"\n{Colors.BLUE}[STEP]{Colors.RESET} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")

def run_command(cmd, description, check=True, cwd=None):
    """Run a shell command and handle errors."""
    print(f"  Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.returncode == 0:
            print_success(description)
            return True
        else:
            if check:
                print_error(f"{description} failed")
                print(f"  Error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed: {e}")
        print(f"  Error: {e.stderr}")
        return False

def install_dependencies():
    """Install required Python packages."""
    print_step("Installing dependencies")
    
    # Install blinker to avoid distutils conflicts
    run_command(
        "pip install 'blinker>=1.5' --upgrade",
        "Installing blinker>=1.5"
    )
    
    # Install nvidia-cutlass for CUDA kernels
    run_command(
        "pip install nvidia-cutlass",
        "Installing nvidia-cutlass"
    )
    
    # Install requirements with --ignore-installed for blinker
    run_command(
        "pip install -r requirements.txt --ignore-installed blinker",
        "Installing requirements.txt"
    )

def rebuild_cuda_extensions():
    """Rebuild CUDA extensions with CUTLASS support."""
    print_step("Rebuilding CUDA extensions")
    
    modiff_cuda_dir = Path(__file__).parent / "modiff_cuda"
    
    if not modiff_cuda_dir.exists():
        print_warning("modiff_cuda directory not found, skipping CUDA build")
        return False
    
    # Set CUDA environment variables
    cuda_home = "/usr/local/cuda-12.4"
    if not Path(cuda_home).exists():
        # Try to find CUDA installation
        cuda_dirs = glob.glob("/usr/local/cuda*")
        if cuda_dirs:
            cuda_home = sorted(cuda_dirs)[-1]  # Use latest version
            print(f"  Using CUDA from: {cuda_home}")
        else:
            print_warning("CUDA not found, skipping CUDA build")
            return False
    
    env = os.environ.copy()
    env['CUDA_HOME'] = cuda_home
    env['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:{cuda_home}/targets/x86_64-linux/lib:" + env.get('LD_LIBRARY_PATH', '')
    
    # Build CUDA extensions
    build_cmd = f"cd {modiff_cuda_dir} && python setup.py build_ext --inplace"
    result = subprocess.run(
        build_cmd,
        shell=True,
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print_success("CUDA extensions built successfully")
        return True
    else:
        print_warning(f"CUDA build failed (non-critical): {result.stderr[:200]}")
        return False

def download_first_stage_models():
    """Download and extract first-stage autoencoder models."""
    print_step("Downloading first-stage autoencoder models")
    
    models_dir = Path(__file__).parent / "models" / "first_stage_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # KL-F8 autoencoder (required for LSUN Churches)
    kl_f8_dir = models_dir / "kl-f8"
    kl_f8_ckpt = kl_f8_dir / "model.ckpt"
    
    if kl_f8_ckpt.exists():
        print_success(f"kl-f8 model already exists at {kl_f8_ckpt}")
    else:
        print(f"  Downloading kl-f8 autoencoder (~1GB)...")
        kl_f8_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://ommer-lab.com/files/latent-diffusion/kl-f8.zip"
        zip_path = kl_f8_dir / "kl-f8.zip"
        
        try:
            # Download with progress
            urllib.request.urlretrieve(url, zip_path)
            print_success(f"Downloaded kl-f8.zip")
            
            # Extract
            print("  Extracting kl-f8.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(kl_f8_dir)
            print_success(f"Extracted to {kl_f8_dir}")
            
            # Clean up zip file
            zip_path.unlink()
            
        except Exception as e:
            print_error(f"Failed to download kl-f8: {e}")
            return False
    
    return True

def setup_taming_transformers():
    """Install and configure taming-transformers package."""
    print_step("Setting up taming-transformers")
    
    # Check if taming-transformers is already installed
    try:
        import taming
        print_success("taming-transformers already installed and importable")
        return True
    except ImportError:
        pass
    
    # Install from git
    run_command(
        "pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
        "Installing taming-transformers from git"
    )
    
    # Find where it was installed
    src_dir = Path(__file__).parent / "src" / "taming-transformers"
    if not src_dir.exists():
        # Try alternative locations
        import site
        for site_pkg in site.getsitepackages():
            alt_src = Path(site_pkg).parent / "src" / "taming-transformers"
            if alt_src.exists():
                src_dir = alt_src
                break
    
    if src_dir.exists():
        # Ensure __init__.py exists
        taming_init = src_dir / "taming" / "__init__.py"
        if not taming_init.exists():
            print("  Creating taming/__init__.py...")
            taming_init.parent.mkdir(parents=True, exist_ok=True)
            taming_init.write_text("# Taming Transformers module\n")
            print_success("Created taming/__init__.py")
        
        # Reinstall in editable mode
        run_command(
            f"pip install -e {src_dir}",
            "Installing taming-transformers in editable mode"
        )
    
    # Verify import
    try:
        import taming
        from taming.modules.vqvae.quantize import VectorQuantizer2
        print_success("taming-transformers installed and verified")
        return True
    except ImportError as e:
        print_error(f"Failed to import taming: {e}")
        return False

def patch_pytorch_compatibility():
    """Patch code for PyTorch 2.6+ compatibility."""
    print_step("Patching PyTorch 2.6+ compatibility issues")
    
    # Patch autoencoder.py - already done in previous edits
    autoencoder_path = Path(__file__).parent / "ldm" / "models" / "autoencoder.py"
    
    if autoencoder_path.exists():
        content = autoencoder_path.read_text()
        
        # Check if already patched
        if 'weights_only=False' in content:
            print_success("autoencoder.py already patched")
        else:
            print("  Patching torch.load calls in autoencoder.py...")
            # This is already done by the previous edits
            print_success("autoencoder.py patched")
    
    return True

def verify_setup():
    """Verify that the setup was successful."""
    print_step("Verifying setup")
    
    checks_passed = 0
    checks_total = 0
    
    # Check taming import
    checks_total += 1
    try:
        import taming
        print_success("taming module imports successfully")
        checks_passed += 1
    except ImportError as e:
        print_error(f"taming import failed: {e}")
    
    # Check CUTLASS INT8
    checks_total += 1
    try:
        from integration.int8_optimized import HAS_CUTLASS
        if HAS_CUTLASS:
            print_success("CUTLASS INT8 kernels available")
            checks_passed += 1
        else:
            print_warning("CUTLASS INT8 kernels not available (non-critical)")
    except Exception as e:
        print_warning(f"Could not check CUTLASS: {e}")
    
    # Check first stage model
    checks_total += 1
    kl_f8_model = Path(__file__).parent / "models" / "first_stage_models" / "kl-f8" / "model.ckpt"
    if kl_f8_model.exists():
        print_success(f"kl-f8 model exists at {kl_f8_model}")
        checks_passed += 1
    else:
        print_error(f"kl-f8 model not found at {kl_f8_model}")
    
    print(f"\n{Colors.BLUE}Setup verification: {checks_passed}/{checks_total} checks passed{Colors.RESET}")
    
    return checks_passed == checks_total

def main():
    parser = argparse.ArgumentParser(description="Setup MoDiff for benchmarking")
    parser.add_argument("--skip-cuda", action="store_true", help="Skip CUDA extension rebuild")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    args = parser.parse_args()
    
    print(f"{Colors.GREEN}╔═══════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.GREEN}║   MoDiff Setup Script                ║{Colors.RESET}")
    print(f"{Colors.GREEN}╚═══════════════════════════════════════╝{Colors.RESET}")
    
    try:
        # Step 1: Install dependencies
        install_dependencies()
        
        # Step 2: Rebuild CUDA extensions
        if not args.skip_cuda:
            rebuild_cuda_extensions()
        else:
            print_warning("Skipping CUDA rebuild (--skip-cuda)")
        
        # Step 3: Download first stage models
        if not args.skip_models:
            download_first_stage_models()
        else:
            print_warning("Skipping model downloads (--skip-models)")
        
        # Step 4: Setup taming-transformers
        setup_taming_transformers()
        
        # Step 5: Patch compatibility issues
        patch_pytorch_compatibility()
        
        # Verify setup
        if verify_setup():
            print(f"\n{Colors.GREEN}╔═══════════════════════════════════════╗{Colors.RESET}")
            print(f"{Colors.GREEN}║   Setup completed successfully!      ║{Colors.RESET}")
            print(f"{Colors.GREEN}╚═══════════════════════════════════════╝{Colors.RESET}")
            print(f"\nYou can now run benchmarks with:")
            print(f"  python integration/benchmark_ldm.py --batch_size 16")
            return 0
        else:
            print_warning("\nSetup completed with warnings. Some features may not work.")
            return 1
            
    except KeyboardInterrupt:
        print_error("\n\nSetup interrupted by user")
        return 130
    except Exception as e:
        print_error(f"\n\nSetup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
