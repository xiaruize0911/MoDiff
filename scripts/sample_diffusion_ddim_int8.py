import argparse, os, glob, datetime, yaml
import logging
import math
import sys
from pathlib import Path

import numpy as np
import tqdm
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

# Add parent directory to path for TensorRT imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_trt_engine(engine_path, device_id=0):
    """
    Load a TensorRT engine for inference.
    
    Args:
        engine_path: Path to the .plan file
        device_id: GPU device ID to use
    
    Returns:
        TRTEngineWrapper instance or None if load fails
    """
    try:
        from trt.inference_wrapper import TRTEngineWrapper
        engine_path = Path(engine_path).expanduser().resolve()
        if not engine_path.exists():
            logger.warning(f"TensorRT engine not found at {engine_path}")
            return None
        logger.info(f"Loading TensorRT engine from {engine_path}")
        wrapper = TRTEngineWrapper(str(engine_path), device=device_id)
        logger.info("TensorRT engine loaded successfully")
        return wrapper
    except Exception as e:
        logger.warning(f"Failed to load TensorRT engine: {e}")
        return None


class TRTInt8SamplingWrapper(nn.Module):
    """
    Wrapper to make TensorRT engine compatible with the sampling pipeline.
    Adds optional output correction using a small FP32 calibration set so
    INT8 TensorRT outputs better align with the original PyTorch model.
    """

    def __init__(self, trt_wrapper):
        super().__init__()
        self.trt_wrapper = trt_wrapper
        self.use_trt = trt_wrapper is not None
        self.device = getattr(trt_wrapper, "torch_device", torch.device("cuda")) if self.use_trt else None
        self.correction_scale = None
        self.correction_bias = None

    def _run_trt(self, x, t):
        if not self.use_trt:
            raise RuntimeError("TensorRT wrapper is None")
        return self.trt_wrapper(x.float(), t.long())

    def calibrate_with_reference(
        self,
        reference_model: nn.Module,
        num_timesteps: int,
        channels: int,
        height: int,
        width: int,
        num_samples: int = 64,
        batch_size: int = 8,
        seed: int | None = None,
    ) -> None:
        if not self.use_trt or reference_model is None or num_samples <= 0:
            return

        logger.info(
            "[TensorRT] Calibrating engine outputs against FP32 reference (%d samples, batch=%d)",
            num_samples,
            batch_size,
        )

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        reference_model.eval()
        reference_model.to(self.device)

        remaining = num_samples
        num_elems = channels * height * width
        scale_num = torch.zeros(num_elems, device=self.device)
        scale_den = torch.zeros(num_elems, device=self.device)
        ref_sum = torch.zeros(num_elems, device=self.device)
        trt_sum = torch.zeros(num_elems, device=self.device)
        count = 0

        with torch.no_grad():
            while remaining > 0:
                cur = 1  # Engine was built with static batch=1
                latent = torch.randn(cur, channels, height, width, device=self.device, generator=generator)
                timesteps = torch.randint(0, num_timesteps, (cur,), device=self.device, generator=generator)

                ref_out = reference_model(latent, timesteps).detach()
                trt_out = self._run_trt(latent, timesteps).detach()

                ref_flat = ref_out.view(cur, -1)
                trt_flat = trt_out.view(cur, -1)

                scale_num += (trt_flat * ref_flat).sum(dim=0)
                scale_den += (trt_flat ** 2).sum(dim=0)
                ref_sum += ref_flat.sum(dim=0)
                trt_sum += trt_flat.sum(dim=0)
                count += cur * num_elems
                remaining -= cur

        eps = 1e-8
        scale = scale_num / (scale_den + eps)
        ref_mean = ref_sum / max(count, 1)
        trt_mean = trt_sum / max(count, 1)
        bias = ref_mean - scale * trt_mean

        scale = torch.clamp(scale, min=0.1, max=2.5)
        self.correction_scale = scale.view(1, channels, height, width)
        self.correction_bias = bias.view(1, channels, height, width)

        logger.info(
            "[TensorRT] Calibration complete: scale mean=%.4f std=%.4f, bias mean=%.4f std=%.4f",
            self.correction_scale.mean().item(),
            self.correction_scale.std().item(),
            self.correction_bias.mean().item(),
            self.correction_bias.std().item(),
        )

    def forward(self, x, t, context=None):
        if not self.use_trt:
            logger.warning("TRTInt8SamplingWrapper: No TensorRT engine loaded, returning zero tensor.")
            return torch.zeros_like(x)

        output = self._run_trt(x, t)
        if self.correction_scale is not None and self.correction_bias is not None:
            output = output * self.correction_scale + self.correction_bias
        return output

    def reset_cache(self):
        pass


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DiffusionINT8(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        config.split_shortcut = self.args.split
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.betas = self.betas.to(self.device)
        betas = self.betas
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        model = Model(self.config)

        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}", root=self.args.model_dir)
        logger.info("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        
        model.to(self.device)
        model.eval()

        if self.args.generate is not None:
            if self.args.generate == 'residual':
                xs, ts, xs_prev, ts_prev = self.generate(model)
                logging.info(f"xs size: {xs.size()}, ts size: {ts.size()}, xs_prev size: {xs_prev.size()}, ts_prev size: {ts_prev.size()}")
                generated_data = {"xs":xs, "ts":ts, "xs_prev":xs_prev, "ts_prev":ts_prev}
            elif self.args.generate == 'raw':
                xs, ts = self.generate(model)
                logging.info(f"xs size: {xs.size()}, ts size: {ts.size()}")
                generated_data = {"xs":xs, "ts":ts}
            else:
                raise ValueError
            torch.save(generated_data, self.args.cali_data_path)
            exit()

        # Try to load TensorRT engine first if specified
        if self.args.use_trt and hasattr(self.args, 'trt_engine_path') and self.args.trt_engine_path:
            logger.info("Attempting to load TensorRT engine for inference")
            device_id = self.device.index if hasattr(self.device, 'index') and self.device.index is not None else 0
            trt_wrapper = load_trt_engine(self.args.trt_engine_path, device_id=device_id)
            if trt_wrapper is not None:
                logger.info("Successfully loaded TensorRT engine, using it for fast inference")
                trt_model = TRTInt8SamplingWrapper(trt_wrapper)
                trt_model.eval()

                calib_samples = max(0, getattr(self.args, "trt_calib_samples", 0))
                if calib_samples > 0:
                    trt_model.calibrate_with_reference(
                        reference_model=model,
                        num_timesteps=self.config.diffusion.num_diffusion_timesteps,
                        channels=self.config.data.channels,
                        height=self.config.data.image_size,
                        width=self.config.data.image_size,
                        num_samples=calib_samples,
                        batch_size=max(1, getattr(self.args, "trt_calib_batch_size", 8)),
                        seed=self.args.seed,
                    )
                    model.cpu()
                    torch.cuda.empty_cache()

                self.sample_fid(trt_model)
                return

        model.to(self.device)
        if self.args.verbose:
            logger.info("quantized model")
            logger.info(model)

        model.eval()
        
        # # test modulation
        # with torch.no_grad():
        #     x = torch.randn((32, 3, 32, 32), device='cuda')
        #     t = torch.ones((32), device='cuda') * 0

        #     import copy
        #     model.reset_cache()
        #     model_copy = copy.deepcopy(model)
        #     for i in range(20):
        #         model_copy.set_quant_state(True, False)
        #         gt = model_copy(x, t)
        #         model.set_quant_state(True, True)
        #         est = model(x, t)
        #         dist_rel = (((((gt-est).view(64,-1))**2).sum(dim=1).sqrt())/((((gt).view(64,-1))**2).sum(dim=1).sqrt())).mean()
        #         print('Relative L2 Distance:', dist_rel)
        #         x = x.clone() + torch.randn((32, 3, 32, 32), device='cuda')*0.01
        #         t = torch.ones((32), device='cuda') * i
        
        # exit()
        if self.args.modulate:
            model.reset_cache()
            
        self.sample_fid(model)
        

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        logger.info(f"starting from image {img_id}")
        total_n_samples = self.args.max_images
        
        # Use batch_size=1 for deterministic generation across different implementations
        # This ensures each image starts with a fresh, deterministic random state
        use_single_image_mode = getattr(self.args, 'deterministic_mode', True)
        
        if use_single_image_mode:
            logger.info("Using single-image mode for deterministic generation")
            n_rounds = total_n_samples - img_id
            batch_size = 1
        else:
            n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)
            batch_size = config.sampling.batch_size

        with torch.no_grad():
            for round_idx in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                # Set seed for each image independently to ensure reproducibility
                if use_single_image_mode:
                    torch.manual_seed(self.args.seed + img_id)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.args.seed + img_id)
                else:
                    if round_idx == 0:
                        torch.manual_seed(self.args.seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(self.args.seed)
                
                n = batch_size if use_single_image_mode else config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                if self.args.modulate:
                    model.reset_cache()

                with torch.amp.autocast('cuda', enabled=False):
                    x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                if img_id + x.shape[0] > self.args.max_images:
                    n = self.args.max_images - img_id
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True, with_t=False):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ddim.functions.denoising import generalized_steps

            betas = self.betas
            deterministic_noise = getattr(self.args, "use_fixed_noise", False)
            base_seed = self.args.seed if deterministic_noise else None
            result = generalized_steps(
                x,
                seq,
                model,
                betas,
                eta=self.args.eta,
                args=self.args,
                with_t=with_t,
                deterministic_noise=deterministic_noise,
                base_seed=base_seed,
            )
            if with_t:
                x, ts, x0_preds = result
            else:
                x, x0_preds = result
        elif self.args.sample_type == "dpm_solver":
            logger.info(f"use dpm-solver with {self.args.timesteps} steps")
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise"
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            return dpm_solver.sample(
                x,
                steps=self.args.timesteps,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
            )
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            deterministic_noise = getattr(self.args, "use_fixed_noise", False)
            base_seed = self.args.seed if deterministic_noise else None
            x = ddpm_steps(
                x,
                seq,
                model,
                self.betas,
                deterministic_noise=deterministic_noise,
                base_seed=base_seed,
            )
        else:
            raise NotImplementedError
        if last:
            x = x[-1]
        return x
    
    def generate(self, model):
        config = self.config
        logger.info(
            "Generating calibration tensors: %d samples across %d timesteps",
            self.args.cali_n,
            self.args.cali_st,
        )
        total_n_samples = self.args.cali_n
        interval = self.args.timesteps // self.args.cali_st
        n_rounds = math.ceil(total_n_samples / config.sampling.batch_size)

        xs_lst = [[] for t in range(self.args.cali_st)]
        ts_lst = [[] for t in range(self.args.cali_st)]
        if self.args.generate == 'residual':
            xs_lst_prev = [[] for t in range(self.args.cali_st)]
            ts_lst_prev = [[] for t in range(self.args.cali_st)]

        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device
                )
                with torch.amp.autocast('cuda', enabled=False):
                    x = self.sample_image(x, model, last=False, with_t=True)
                    for t in range(len(x[1])):
                        if t % interval == 0:
                            xs_lst[t // interval].append(x[0][t])
                            ts_lst[t // interval].append(x[1][t])
                            if self.args.generate == 'residual':
                                if t <= 1:
                                    xs_lst_prev[t // interval].append(x[0][t])
                                    ts_lst_prev[t // interval].append(x[1][t])
                                else:
                                    xs_lst_prev[t // interval].append(x[0][t-1])
                                    ts_lst_prev[t // interval].append(x[1][t-1])
                
        xs = []
        for item in xs_lst:
            for idx in range(len(item)):
                item[idx] = item[idx].cpu()
            xs.append(torch.cat(item, dim=0))
        xs = torch.stack(xs, dim=0)

        ts = []
        for item in ts_lst:
            for idx in range(len(item)):
                item[idx] = item[idx].cpu()
            ts.append(torch.cat(item, dim=0))
        ts = torch.stack(ts, dim=0)

        if self.args.generate == 'residual':
            xs_prev = []
            for item in xs_lst_prev:
                for idx in range(len(item)):
                    item[idx] = item[idx].cpu()
                xs_prev.append(torch.cat(item, dim=0))
            xs_prev = torch.stack(xs_prev, dim=0)

            ts_prev = []
            for item in ts_lst_prev:
                for idx in range(len(item)):
                    item[idx] = item[idx].cpu()
                ts_prev.append(torch.cat(item, dim=0))
            ts_prev = torch.stack(ts_prev, dim=0)

            return xs, ts, xs_prev, ts_prev
        else:
            return xs, ts


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Path to the model directory"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--max_images", type=int, default=50000, help="number of images to sample"
    )

    # Calibration data generation (optional)
    parser.add_argument(
        "--cali_st", type=int, default=1,
        help="timesteps captured when exporting calibration data"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024,
        help="number of samples stored per timestep during calibration export"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )

    # MoDiff parameters
    parser.add_argument("--modulate", action="store_true", help="if apply modulated computing")
    parser.add_argument("--generate", type=str, default=None, choices=[None, "raw", "residual"], help="generate calibration data")

    # TensorRT parameters
    parser.add_argument("--use_trt", action="store_true", help="use TensorRT engine for inference acceleration")
    parser.add_argument("--trt_engine_path", type=str, default=None, help="path to TensorRT engine (.plan file)")
    parser.add_argument(
        "--trt_calib_samples",
        type=int,
        default=64,
        help="number of FP32 reference samples to fit TensorRT output correction",
    )
    parser.add_argument(
        "--trt_calib_batch_size",
        type=int,
        default=8,
        help="batch size used during TensorRT output calibration",
    )
    parser.add_argument(
        "--use_fixed_noise",
        action="store_true",
        help="reuse a fixed noise tensor for deterministic sampling",
    )

    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    logdir = os.path.join(args.logdir, "samples")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir

    if not os.path.exists(imglogdir):
        os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    runner = DiffusionINT8(args, config)
    runner.sample()
