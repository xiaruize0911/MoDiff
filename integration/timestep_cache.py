"""
Cached Timestep Embeddings for Diffusion Models.

Timestep embeddings are computed via expensive sinusoidal functions (sin/cos) 
which are deterministic given the timestep value. By caching these embeddings, 
we eliminate redundant computation.

Expected speedup: 2-3% (timestep embedding accounts for ~2-5% of forward pass time)

Usage:
    from integration.timestep_cache import get_cached_timestep_embedding
    
    # In UNet forward():
    # OLD: t_emb = timestep_embedding(timesteps, self.model_channels)
    # NEW: t_emb = get_cached_timestep_embedding(timesteps, self.model_channels)
"""

import torch
import math
from typing import Dict, Tuple, Optional
from ldm.modules.diffusionmodules.util import timestep_embedding as _original_timestep_embedding


class TimestepEmbeddingCache:
    """
    Global cache for timestep embeddings.
    
    Caches the expensive sinusoidal embedding computation per unique (timestep, dim) pair.
    Since diffusion models use fixed timestep schedules (e.g., 50 or 1000 steps),
    we can precompute and reuse these embeddings.
    
    Memory cost: Negligible (~50 timesteps × 1024 dims × 4 bytes = 200KB)
    Speedup: 2-3% by eliminating sin/cos computation overhead
    """
    
    def __init__(self):
        # Cache: (dim, max_period) -> Dict[timestep_tuple -> embedding_tensor]
        self.cache: Dict[Tuple[int, int], Dict[tuple, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0
        
    def get_embedding(
        self,
        timesteps: torch.Tensor,
        dim: int,
        max_period: int = 10000,
        repeat_only: bool = False
    ) -> torch.Tensor:
        """
        Get cached timestep embedding or compute and cache it.
        
        Args:
            timesteps: 1-D tensor of timestep indices
            dim: Embedding dimension
            max_period: Maximum period for sinusoidal embedding
            repeat_only: If True, just repeat timesteps instead of sinusoidal
            
        Returns:
            Tensor of shape [N, dim] with timestep embeddings
        """
        if timesteps.is_cuda and torch.cuda.is_current_stream_capturing():
            return _original_timestep_embedding(timesteps, dim, max_period, repeat_only)

        # Create cache key for this configuration
        config_key = (dim, max_period, repeat_only)
        
        if config_key not in self.cache:
            self.cache[config_key] = {}
        
        config_cache = self.cache[config_key]
        
        # Convert timesteps to hashable key
        # For batch of timesteps, we cache each unique value separately
        timesteps_cpu = timesteps.cpu()
        device = timesteps.device
        dtype = timesteps.dtype
        
        # Check if all timesteps are in cache
        all_cached = True
        timestep_keys = []
        for t in timesteps_cpu:
            t_key = tuple(t.tolist()) if t.numel() > 1 else (t.item(),)
            timestep_keys.append(t_key)
            if t_key not in config_cache:
                all_cached = False
                break
        
        if all_cached:
            # Fast path: retrieve from cache
            self.hits += 1
            embeddings = [config_cache[tk].to(device) for tk in timestep_keys]
            return torch.cat(embeddings, dim=0) if len(embeddings) > 1 else embeddings[0]
        
        # Slow path: compute embedding
        self.misses += 1
        embedding = _original_timestep_embedding(timesteps, dim, max_period, repeat_only)
        
        # Cache individual timestep embeddings
        for i, t_key in enumerate(timestep_keys):
            if t_key not in config_cache:
                # Store on CPU to save GPU memory
                config_cache[t_key] = embedding[i:i+1].cpu().clone()
        
        return embedding
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_entries = sum(len(cache) for cache in self.cache.values())
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0,
            'total_entries': total_entries,
            'configs': len(self.cache)
        }


# Global cache instance
_global_timestep_cache = None

def get_timestep_cache() -> TimestepEmbeddingCache:
    """Get the global timestep embedding cache."""
    global _global_timestep_cache
    if _global_timestep_cache is None:
        _global_timestep_cache = TimestepEmbeddingCache()
    return _global_timestep_cache


def get_cached_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False
) -> torch.Tensor:
    """
    Get timestep embedding with caching.
    
    Drop-in replacement for timestep_embedding() from util.py.
    
    Args:
        timesteps: 1-D tensor of timestep indices
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal embedding
        repeat_only: If True, just repeat timesteps
        
    Returns:
        Tensor of shape [N, dim] with timestep embeddings
    """
    cache = get_timestep_cache()
    return cache.get_embedding(timesteps, dim, max_period, repeat_only)


def clear_timestep_cache():
    """Clear the global timestep embedding cache."""
    cache = get_timestep_cache()
    cache.clear()


def print_timestep_cache_stats():
    """Print cache statistics."""
    cache = get_timestep_cache()
    stats = cache.get_stats()
    print(f"Timestep Embedding Cache Stats:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    print(f"  Cached Entries: {stats['total_entries']}")
    print(f"  Configurations: {stats['configs']}")
