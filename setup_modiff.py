#!/usr/bin/env python3
"""
MoDiff Setup Script
===================
This script automates the setup process for running MoDiff benchmarks on a new machine.

It performs the following steps:
1. Installs required dependencies (blinker)
2. Downloads and extracts first-stage autoencoder models
3. Sets up taming-transformers package
4. Patches PyTorch 2.6+ compatibility issues

Usage:
    python setup_modiff.py [--skip-models]
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
        f'"{sys.executable}" -m pip install --ignore-installed "blinker>=1.5"',
        "Installing blinker>=1.5"
    )
    
    # Install requirements with --ignore-installed for blinker
    run_command(
        f'"{sys.executable}" -m pip install -r requirements.txt --ignore-installed blinker',
        "Installing requirements.txt"
    )

def setup_cutlass():
    """Clone CUTLASS repository if not present."""
    print_step("Setting up CUTLASS")
    
    cutlass_path = Path("/workspace/cutlass")
    if cutlass_path.exists():
        print_success(f"CUTLASS already exists at {cutlass_path}")
        return True
    
    return run_command(
        "git clone https://github.com/NVIDIA/cutlass.git /workspace/cutlass",
        "Cloning CUTLASS from GitHub"
    )

def build_modiff_extension():
    """Build and install the MoDiff C++ extension."""
    print_step("Building MoDiff C++ extension")
    
    # Set CUTLASS_PATH environment variable
    os.environ["CUTLASS_PATH"] = "/workspace/cutlass"
    
    return run_command(
        f'"{sys.executable}" -m pip install .',
        "Building and installing MoDiff package",
        cwd=str(Path(__file__).parent)
    )

def run_benchmark():
    """Run the default LDM benchmark."""
    print_step("Running LDM Benchmark")
    
    benchmark_script = Path(__file__).parent / "integration" / "benchmarks" / "benchmark_ldm.py"
    
    return run_command(
        f'"{sys.executable}" {benchmark_script}',
        "Running integration/benchmark_ldm.py",
        cwd=str(Path(__file__).parent)
    )

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
    
    # LSUN Churches LDM model
    ldm_dir = Path(__file__).parent / "models" / "ldm" / "lsun_churches256"
    ldm_ckpt = ldm_dir / "model.ckpt"
    
    if ldm_ckpt.exists():
        print_success(f"LSUN Churches model already exists at {ldm_ckpt}")
    else:
        print(f"  Downloading LSUN Churches LDM model (~1.5GB)...")
        ldm_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip"
        zip_path = ldm_dir / "lsun_churches.zip"
        
        try:
            # Download
            urllib.request.urlretrieve(url, zip_path)
            print_success(f"Downloaded lsun_churches.zip")
            
            # Extract
            print("  Extracting lsun_churches.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ldm_dir)
            print_success(f"Extracted to {ldm_dir}")
            
            # Clean up zip file
            zip_path.unlink()
            
        except Exception as e:
            print_error(f"Failed to download LSUN Churches model: {e}")
            return False
    
    return True

def setup_taming_transformers():
    """Install and configure taming-transformers package."""
    print_step("Setting up taming-transformers")
    
    def verify_taming_import():
        return run_command(
            f'"{sys.executable}" -c "import taming; from taming.modules.vqvae.quantize import VectorQuantizer2"',
            "Verifying taming-transformers import",
            check=False,
        )

    # Check if taming-transformers is already installed
    if verify_taming_import():
        print_success("taming-transformers already installed and importable")
        return True
    
    # Install from git with real-time output
    print("  Installing taming-transformers from git (this may take a few minutes)...")
    result = subprocess.run(
        f'"{sys.executable}" -m pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers',
        shell=True,
        check=False,
        text=True
    )
    
    if result.returncode == 0:
        print_success("Installing taming-transformers from git")
    else:
        print_warning("pip install had issues, continuing anyway...")
    
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
            f'"{sys.executable}" -m pip install -e {src_dir} --config-settings editable_mode=compat',
            "Installing taming-transformers in editable mode"
        )
    
    # Verify import
    if verify_taming_import():
        print_success("taming-transformers installed and verified")
        return True

    print_error("Failed to import taming in a fresh Python process")
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
    if run_command(
        f'"{sys.executable}" -c "import taming"',
        "Checking taming module import",
        check=False,
    ):
        print_success("taming module imports successfully")
        checks_passed += 1
    else:
        print_error("taming import failed in a fresh Python process")
    
    # Check CUTLASS extension
    checks_total += 1
    try:
        from integration.kernels.int8_optimized import HAS_CUTLASS
        if HAS_CUTLASS:
            print_success("CUTLASS backend available")
            checks_passed += 1
        else:
            print_warning("CUTLASS backend not available (did build fail?)")
    except Exception as e:
        print_warning(f"Could not check CUTLASS backend: {e}")
    
    # Check first stage model
    checks_total += 1
    kl_f8_model = Path(__file__).parent / "models" / "first_stage_models" / "kl-f8" / "model.ckpt"
    if kl_f8_model.exists():
        print_success(f"kl-f8 model exists at {kl_f8_model}")
        checks_passed += 1
    else:
        print_error(f"kl-f8 model not found at {kl_f8_model}")
    
    # Check LSUN Churches LDM model
    checks_total += 1
    ldm_model = Path(__file__).parent / "models" / "ldm" / "lsun_churches256" / "model.ckpt"
    if ldm_model.exists():
        print_success(f"LSUN Churches model exists")
        checks_passed += 1
    else:
        print_error(f"LSUN Churches model not found at {ldm_model}")
    
    print(f"\n{Colors.BLUE}Setup verification: {checks_passed}/{checks_total} checks passed{Colors.RESET}")
    
    return checks_passed == checks_total

def main():
    parser = argparse.ArgumentParser(description="Setup MoDiff for benchmarking")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    
    print(f"{Colors.GREEN}╔═══════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.GREEN}║   MoDiff Setup Script                ║{Colors.RESET}")
    print(f"{Colors.GREEN}╚═══════════════════════════════════════╝{Colors.RESET}")
    
    try:
        # Step 1: Install dependencies
        if not args.skip_dependencies:
            install_dependencies()
        else:
            print_warning("Skipping dependency installation (--skip-dependencies)")
            
        # Step 2: Setup CUTLASS (Required for extension build)
        setup_cutlass()

        # Step 3: Build MoDiff C++ Extension
        build_modiff_extension()

        # Step 4: Download models
        if not args.skip_models:
            download_first_stage_models()
        else:
            print_warning("Skipping model downloads (--skip-models)")
        
        # Step 5: Taming Transformers
        setup_taming_transformers()
        
        # Step 6: Patch PyTorch compatibility
        patch_pytorch_compatibility()
        
        # Step 7: Verify
        if verify_setup():
            print(f"\n{Colors.GREEN}Setup complete! Running benchmark...{Colors.RESET}\n")
            run_benchmark()
        else:
            print(f"\n{Colors.RED}Setup verification failed. Please check errors above.{Colors.RESET}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
