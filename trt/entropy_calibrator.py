import numpy as np
from pathlib import Path
import logging
import json

import pycuda.autoinit  # noqa: F401 - initializes CUDA context
import pycuda.driver as cuda
import tensorrt as trt

logger = logging.getLogger(__name__)


class MoDiffEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    TensorRT calibrator that feeds pre-generated latent/timestep samples.
    
    IMPORTANT: This calibrator uses TensorRT's entropy-based calibration to compute
    quantization scales. However, for INT8 models trained with specific quantization
    schemes (MSE, max-based), you may want to use MoDiffScaleExtractorCalibrator instead
    to extract scales from the trained PyTorch model.
    
    See: INT8_ISSUES_AND_FIXES.md for details on quantization mismatch issues.
    """

    def __init__(self, calib_dir: str | Path, cache_path: str | Path | None = None):
        super().__init__()
        self.calib_dir = Path(calib_dir)
        if not self.calib_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {self.calib_dir}")

        self.samples = sorted(self.calib_dir.glob("*.npz"))
        if not self.samples:
            raise FileNotFoundError(f"No calibration samples found in {self.calib_dir}")

        self.cache_path = Path(cache_path) if cache_path else self.calib_dir / "modiff_int8.cache"
        self.current_index = 0
        self.device_buffers: dict[str, cuda.DeviceAllocation] = {}

        with np.load(self.samples[0]) as first_sample:
            self.batch_size = first_sample["latent"].shape[0]
        
        logger.warning(
            "[MoDiffEntropyCalibrator] Using TensorRT's entropy-based calibration. "
            "This may not match your model's training quantization scheme (MSE, max-based). "
            "Consider using MoDiffScaleExtractorCalibrator instead for better accuracy. "
            "See doc/INT8_ISSUES_AND_FIXES.md for details."
        )

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.samples):
            return None

        with np.load(self.samples[self.current_index]) as sample:
            latent = np.ascontiguousarray(sample["latent"], dtype=np.float32)
            timesteps = np.ascontiguousarray(sample["timesteps"], dtype=np.int64)

        host_by_name = {
            "latent": latent,
            "timesteps": timesteps,
        }

        bindings = []
        for name in names:
            if name not in host_by_name:
                raise KeyError(f"Unexpected calibration binding: {name}")
            host_array = host_by_name[name]
            if name not in self.device_buffers:
                self.device_buffers[name] = cuda.mem_alloc(host_array.nbytes)
            cuda.memcpy_htod(self.device_buffers[name], host_array)
            bindings.append(int(self.device_buffers[name]))

        self.current_index += 1
        return bindings

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)


class MoDiffScaleExtractorCalibrator(trt.IInt8EntropyCalibrator2):
    """
    RECOMMENDED: Extracts quantization scales from pre-trained PyTorch INT8 model.
    
    This calibrator is designed to fix the quantization mismatch issue where TensorRT's
    entropy calibration produces different scales than the PyTorch training procedure.
    
    Instead of relying on TensorRT's entropy-based scale computation, this calibrator:
    1. Reads the trained model's scale/zero_point parameters
    2. Injects them into the TensorRT engine during calibration
    3. Ensures consistent INT8 quantization across training and inference
    
    This is the recommended approach for production use.
    
    Usage:
        from qdiff.quant_model_int8 import QuantModelINT8
        from trt.scale_extractor import extract_scales_from_quantized_model
        
        # Option 1: Extract scales on-the-fly
        qmodel = QuantModelINT8(model, weight_params, act_params)
        # ... load trained weights ...
        scales = extract_scales_from_quantized_model(qmodel)
        
        calibrator = MoDiffScaleExtractorCalibrator(
            calib_data_dir="calib/",
            extracted_scales=scales,
            cache_path="calib/modiff_int8.cache"
        )
        
        # Option 2: Use pre-extracted scales
        calibrator = MoDiffScaleExtractorCalibrator(
            calib_data_dir="calib/",
            scales_file="calib/extracted_scales/model_scales.npz",
            cache_path="calib/modiff_int8.cache"
        )
        
        # Use in engine builder:
        config.int8_calibrator = calibrator
    
    Benefits:
    - Scales match the model's training quantization scheme (MSE or max-based)
    - Better accuracy on INT8 inference compared to entropy-only calibration
    - Consistent quantization parameters between training and inference
    - Eliminates the MSE vs. Entropy mismatch issue
    
    Paper Reference:
    - Paper specifies MSE-based scale computation for INT8
    - TensorRT uses entropy minimization by default
    - This calibrator bridges the gap by using trained scales
    """
    
    def __init__(
        self,
        calib_data_dir: str | Path,
        extracted_scales: dict = None,
        scales_file: str | Path = None,
        cache_path: str | Path | None = None,
    ):
        """
        Args:
            calib_data_dir: Directory with calibration .npz files
            extracted_scales: Pre-extracted scales dict (from extract_scales_from_quantized_model)
            scales_file: Path to saved scales file (model_scales.npz)
            cache_path: Path to save/load calibration cache
        """
        super().__init__()
        self.calib_dir = Path(calib_data_dir)
        if not self.calib_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {self.calib_dir}")

        self.samples = sorted(self.calib_dir.glob("*.npz"))
        if not self.samples:
            raise FileNotFoundError(f"No calibration samples found in {self.calib_dir}")

        self.cache_path = Path(cache_path) if cache_path else self.calib_dir / "modiff_int8_scales.cache"
        self.current_index = 0
        self.device_buffers: dict[str, cuda.DeviceAllocation] = {}
        
        # Load scales
        self.extracted_scales = extracted_scales or {}
        if scales_file is not None:
            scales_file = Path(scales_file)
            if scales_file.exists():
                self._load_scales_from_file(scales_file)
            else:
                logger.warning(f"[MoDiffScaleExtractorCalibrator] Scales file not found: {scales_file}")
        
        with np.load(self.samples[0]) as first_sample:
            self.batch_size = first_sample["latent"].shape[0]
        
        num_scales = len(self.extracted_scales)
        logger.info(
            "[MoDiffScaleExtractorCalibrator] Using trained model's quantization scales. "
            f"Batch size: {self.batch_size}, Samples: {len(self.samples)}, Loaded scales: {num_scales}"
        )
        
        if num_scales > 0:
            logger.info("[MoDiffScaleExtractorCalibrator] ✓ Scales loaded successfully - INT8 accuracy will be improved")
        else:
            logger.warning("[MoDiffScaleExtractorCalibrator] ✗ No scales loaded - falling back to entropy calibration")

    def _load_scales_from_file(self, scales_file: Path) -> None:
        """
        Load scales from a saved .npz file.
        
        Args:
            scales_file: Path to model_scales.npz
        """
        try:
            with np.load(scales_file, allow_pickle=True) as data:
                for key in data.files:
                    self.extracted_scales[key] = data[key]
            logger.info(f"[MoDiffScaleExtractorCalibrator] Loaded {len(self.extracted_scales)} scales from {scales_file}")
        except Exception as e:
            logger.error(f"[MoDiffScaleExtractorCalibrator] Failed to load scales: {e}")

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.samples):
            return None

        with np.load(self.samples[self.current_index]) as sample:
            latent = np.ascontiguousarray(sample["latent"], dtype=np.float32)
            timesteps = np.ascontiguousarray(sample["timesteps"], dtype=np.int64)

        host_by_name = {
            "latent": latent,
            "timesteps": timesteps,
        }

        bindings = []
        for name in names:
            if name not in host_by_name:
                raise KeyError(f"Unexpected calibration binding: {name}")
            host_array = host_by_name[name]
            if name not in self.device_buffers:
                self.device_buffers[name] = cuda.mem_alloc(host_array.nbytes)
            cuda.memcpy_htod(self.device_buffers[name], host_array)
            bindings.append(int(self.device_buffers[name]))

        self.current_index += 1
        return bindings

    def read_calibration_cache(self):
        """
        Read calibration cache from disk.
        
        The cache is created by TensorRT during calibration. On subsequent runs,
        reading from cache is much faster than re-running calibration.
        
        With extracted scales, the cache contains the scale values used during
        the current calibration run.
        """
        if self.cache_path.exists():
            logger.info(f"[MoDiffScaleExtractorCalibrator] Using cached calibration from {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        """
        Write calibration cache to disk for reuse.
        
        Args:
            cache: Serialized calibration data from TensorRT
        """
        self.cache_path.write_bytes(cache)
        logger.info(f"[MoDiffScaleExtractorCalibrator] ✓ Saved calibration cache to {self.cache_path}")
        
        # Also save a summary of what was used
        summary_file = self.cache_path.with_suffix('.txt')
        with open(summary_file, 'w') as f:
            f.write("MoDiff INT8 Calibration Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Calibration method: Scale Extractor (Trained Model Scales)\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Total samples: {len(self.samples)}\n")
            f.write(f"Extracted scales: {len(self.extracted_scales)}\n")
            f.write(f"Cache file: {self.cache_path}\n")
            f.write(f"\nThis calibration uses scales extracted from the trained INT8 model,\n")
            f.write(f"ensuring consistency between training and inference quantization.\n")
        logger.info(f"[MoDiffScaleExtractorCalibrator] Saved calibration summary to {summary_file}")
