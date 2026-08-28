"""Global CUDA-graph phase: pin residual replay on or off for a static capture.

None  — eager cadence (step_count % MODIFF_REPLAY_K)
True  — every eligible conv takes the residual-replay path (no GN/conv)
False — every conv takes the full compute path
"""
from typing import Optional

_FORCE_REPLAY: Optional[bool] = None


def set_force_replay(value: Optional[bool]) -> None:
    global _FORCE_REPLAY
    _FORCE_REPLAY = value


def get_force_replay() -> Optional[bool]:
    return _FORCE_REPLAY
