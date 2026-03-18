"""
Infrastructure and utility modules.

Modules:
    buffer_pool      - Pre-allocated GPU buffer pool for reduced allocation overhead
    timestep_cache   - Cached timestep embeddings to avoid recomputation
    profiler         - CUDA event-based profiling for kernel and pipeline timing
"""
