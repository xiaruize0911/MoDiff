import argparse
from pathlib import Path
import yaml
import torch

from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path


def dict_to_namespace(config_dict):
	namespace = argparse.Namespace()
	for key, value in config_dict.items():
		if isinstance(value, dict):
			setattr(namespace, key, dict_to_namespace(value))
		else:
			setattr(namespace, key, value)
	return namespace


def load_cifar10_model(config_path: Path, ckpt_root: str | None) -> tuple[Model, argparse.Namespace]:
	with open(config_path, "r", encoding="utf-8") as handle:
		raw_config = yaml.safe_load(handle)
	config = dict_to_namespace(raw_config)
	config.split_shortcut = False

	model = Model(config)

	if config.data.dataset != "CIFAR10":
		raise ValueError(f"Expected CIFAR10 dataset in config, found {config.data.dataset}")

	ckpt_path = get_ckpt_path("ema_cifar10", root=ckpt_root)
	state_dict = torch.load(ckpt_path, map_location="cpu")
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]
	model.load_state_dict(state_dict, strict=True)
	return model, config


def export_unet_onnx(model: Model, config: argparse.Namespace, output_path: Path, device: torch.device, opset: int) -> None:
	model.eval()
	model.to(device)

	latent = torch.randn(
		1,
		config.data.channels,
		config.data.image_size,
		config.data.image_size,
		device=device,
	)
	timesteps = torch.randint(
		low=0,
		high=config.diffusion.num_diffusion_timesteps,
		size=(1,),
		device=device,
		dtype=torch.long,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.onnx.export(
		model,
		(latent, timesteps),
		output_path.as_posix(),
		export_params=True,
		opset_version=opset,
		input_names=["latent", "timesteps"],
		output_names=["noise_pred"],
		dynamic_axes={"latent": {0: "batch"}, "timesteps": {0: "batch"}, "noise_pred": {0: "batch"}},
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export MoDiff CIFAR10 UNet to ONNX")
	parser.add_argument(
		"--config",
		default=str(Path(__file__).resolve().parent.parent / "configs" / "cifar10.yml"),
		help="Path to CIFAR10 YAML config",
	)
	parser.add_argument(
		"--output",
		default=str(Path.cwd() / "modiff_unet_cifar10.onnx"),
		help="Destination ONNX file",
	)
	parser.add_argument(
		"--ckpt-root",
		default=None,
		help="Optional root directory for pretrained checkpoints",
	)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device used for export",
	)
	parser.add_argument(
		"--opset",
		type=int,
		default=17,
		help="ONNX opset version",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config_path = Path(args.config).expanduser().resolve()
	output_path = Path(args.output).expanduser().resolve()
	device = torch.device(args.device)

	model, config = load_cifar10_model(config_path, args.ckpt_root)
	export_unet_onnx(model, config, output_path, device, args.opset)
	print(f"Exported UNet to {output_path}")


if __name__ == "__main__":
	main()