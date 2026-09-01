"""CUDA-graph phase pin. Residual-replay (scheme B/C) was removed.

Previously `True` forced every conv onto the skip-GN+conv path. Capture now
only records `first` and `modulated`; these helpers always report None.
"""
from typing import Optional

_FORCE_REPLAY: Optional[bool] = None


def set_force_replay(value: Optional[bool]) -> None:
    global _FORCE_REPLAY
    _FORCE_REPLAY = None


def get_force_replay() -> Optional[bool]:
    return None
