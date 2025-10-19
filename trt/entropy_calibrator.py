import numpy as np
from pathlib import Path

import pycuda.autoinit  # noqa: F401 - initializes CUDA context
import pycuda.driver as cuda
import tensorrt as trt


class MoDiffEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT calibrator that feeds pre-generated latent/timestep samples."""

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
