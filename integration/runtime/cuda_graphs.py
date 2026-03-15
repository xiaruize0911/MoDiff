from __future__ import annotations

import torch
from ldm.models.diffusion import ddim as ddim_module


class SamplerCudaGraphRunner:
    """Capture and replay a fixed-shape DDIM sampling pass with CUDA Graphs."""

    def __init__(self, sampler, steps: int, batch_size: int, shape: tuple,
                 use_autocast: bool = False, dtype: torch.dtype | None = None,
                 warmup_iters: int = 2):
        self.sampler = sampler
        self.steps = steps
        self.batch_size = batch_size
        self.shape = shape
        self.use_autocast = use_autocast
        self.dtype = dtype
        self.warmup_iters = warmup_iters
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_samples = None
        self.static_intermediates = None
        self.static_x_T = None

    def _run_sampling(self):
        sample_shape = (self.batch_size, *self.shape)
        self.static_samples, self.static_intermediates = self.sampler.ddim_sampling(
            cond=None,
            shape=sample_shape,
            x_T=self.static_x_T,
            ddim_use_original_steps=False,
            log_every_t=self.steps + 1,
        )

    def capture(self):
        self.sampler.make_schedule(ddim_num_steps=self.steps, ddim_eta=0.0, verbose=False)
        self.static_x_T = torch.randn((self.batch_size, *self.shape), device=self.sampler.model.device)

        original_tqdm = ddim_module.tqdm
        ddim_module.tqdm = lambda iterable, **kwargs: iterable
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(warmup_stream):
                for _ in range(self.warmup_iters):
                    with torch.inference_mode(), torch.amp.autocast(
                        'cuda', enabled=self.use_autocast, dtype=self.dtype
                    ):
                        self._run_sampling()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                with torch.inference_mode(), torch.amp.autocast(
                    'cuda', enabled=self.use_autocast, dtype=self.dtype
                ):
                    self._run_sampling()
            torch.cuda.synchronize()
        finally:
            ddim_module.tqdm = original_tqdm

    def replay(self):
        if self.graph is None:
            raise RuntimeError('CUDA graph has not been captured yet')
        self.graph.replay()
        return self.static_samples, self.static_intermediates