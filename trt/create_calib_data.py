import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path


def dict_to_namespace(data):
    namespace = argparse.Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def load_cifar_config(config_path: Path) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return dict_to_namespace(raw_config)


def load_pretrained_model(config: argparse.Namespace, ckpt_root: str | None) -> Model:
    if config.data.dataset != "CIFAR10":
        raise ValueError(f"Expected CIFAR10 dataset in config, found {config.data.dataset}")

    model = Model(config)
    ckpt_path = get_ckpt_path("ema_cifar10", root=ckpt_root)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def generate_calibration_samples(
    num_samples: int,
    batch_size: int,
    channels: int,
    image_size: int,
    num_timesteps: int,
    device: torch.device,
    context_dim: int = None,
):
    remaining = num_samples
    while remaining > 0:
        current_bs = min(batch_size, remaining)
        latents = torch.randn(current_bs, channels, image_size, image_size, device=device)
        timesteps = torch.randint(
            low=0,
            high=num_timesteps,
            size=(current_bs,),
            device=device,
            dtype=torch.long,
        )
        
        batch_data = {
            "latent": latents.cpu().numpy().astype(np.float32),
            "timesteps": timesteps.cpu().numpy().astype(np.int64),
        }
        
        if context_dim is not None:
            # Standard sequence length for CLIP text encoder is 77
            seq_len = 77
            context = torch.randn(current_bs, seq_len, context_dim, device=device)
            batch_data["context"] = context.cpu().numpy().astype(np.float32)
            
        yield batch_data
        remaining -= current_bs


def save_samples_to_dir(output_dir: Path, iterator) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, batch_data in enumerate(iterator):
        filename = output_dir / f"sample_{index:04d}.npz"
        np.savez_compressed(filename, **batch_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create calibration data for MoDiff UNet INT8 calibration")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "configs" / "cifar10.yml"),
        help="Path to CIFAR10 YAML config",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "calibration"),
        help="Directory to store generated calibration samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Total number of latent samples to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for random latent generation",
    )
    parser.add_argument(
        "--ckpt-root",
        default=None,
        help="Optional directory containing pretrained checkpoints",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    device = torch.device(args.device)

    config = load_cifar_config(config_path)
    
    # Try to load model if it's CIFAR10, just to verify config
    try:
        if hasattr(config, "data") and config.data.dataset == "CIFAR10":
            _ = load_pretrained_model(config, args.ckpt_root)
    except Exception as e:
        print(f"Warning: Could not load pretrained model: {e}")

    # Extract parameters
    channels = 3
    image_size = 32
    num_timesteps = 1000
    context_dim = None

    # CIFAR10 / DDIM structure
    if hasattr(config, "data"):
        channels = getattr(config.data, "channels", channels)
        image_size = getattr(config.data, "image_size", image_size)
    if hasattr(config, "diffusion"):
        num_timesteps = getattr(config.diffusion, "num_diffusion_timesteps", num_timesteps)

    # LDM / Stable Diffusion structure
    if hasattr(config, "model") and hasattr(config.model, "params"):
        params = config.model.params
        if hasattr(params, "timesteps"):
            num_timesteps = params.timesteps
        
        if hasattr(params, "unet_config") and hasattr(params.unet_config, "params"):
            unet_params = params.unet_config.params
            if hasattr(unet_params, "in_channels"):
                channels = unet_params.in_channels
            if hasattr(unet_params, "image_size"):
                # LDM usually operates in latent space, so image_size might be latent size
                # But config might say 32 or 64.
                image_size = unet_params.image_size
            if hasattr(unet_params, "context_dim"):
                context_dim = unet_params.context_dim
                print(f"Detected LDM/SD config, using context_dim={context_dim}")

    print(f"Generating calibration data: {args.num_samples} samples")
    print(f"  Shape: {channels}x{image_size}x{image_size}")
    print(f"  Timesteps: {num_timesteps}")
    if context_dim:
        print(f"  Context dim: {context_dim}")

    generators = generate_calibration_samples(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        channels=channels,
        image_size=image_size,
        num_timesteps=num_timesteps,
        device=device,
        context_dim=context_dim,
    )

    save_samples_to_dir(output_dir, generators)
    print(f"Saved calibration data to {output_dir}")


if __name__ == "__main__":
    main()