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
        yield latents.cpu().numpy().astype(np.float32), timesteps.cpu().numpy().astype(np.int64)
        remaining -= current_bs


def save_samples_to_dir(output_dir: Path, iterator) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (latent_batch, timestep_batch) in enumerate(iterator):
        filename = output_dir / f"sample_{index:04d}.npz"
        np.savez_compressed(filename, latent=latent_batch, timesteps=timestep_batch)


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
    _ = load_pretrained_model(config, args.ckpt_root)

    generators = generate_calibration_samples(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        channels=config.data.channels,
        image_size=config.data.image_size,
        num_timesteps=config.diffusion.num_diffusion_timesteps,
        device=device,
    )

    save_samples_to_dir(output_dir, generators)
    print(f"Saved calibration data to {output_dir}")


if __name__ == "__main__":
    main()