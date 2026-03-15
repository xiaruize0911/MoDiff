import time
import torch
from collections import defaultdict
import threading

class Profiler:
    """
    Zero-overhead profiler using CUDA events.
    
    When enabled=True, uses asynchronous CUDA events for timing (no GPU sync).
    When enabled=False, start/stop are complete no-ops.
    Call collect() or print_summary() to synchronize and gather results.
    """
    _events = defaultdict(list)  # name -> list of (start_event, end_event)
    _stats = defaultdict(float)
    _counts = defaultdict(int)
    _collected = False
    enabled = False  # Disabled by default for zero overhead
    
    @classmethod
    def reset(cls):
        cls._stats.clear()
        cls._counts.clear()
        cls._events.clear()
        cls._collected = False
    
    @classmethod
    def start(cls, name):
        if not cls.enabled:
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev
        
    @classmethod
    def stop(cls, name, start_event):
        if not cls.enabled or start_event is None:
            return
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        cls._events[name].append((start_event, end_ev))

    @classmethod
    def collect(cls):
        """Synchronize GPU and collect all event timings. Call before print_summary."""
        if cls._collected:
            return
        torch.cuda.synchronize()
        for name, pairs in cls._events.items():
            total = 0.0
            for start_ev, end_ev in pairs:
                total += start_ev.elapsed_time(end_ev)  # milliseconds
            cls._stats[name] += total / 1000.0  # convert to seconds
            cls._counts[name] += len(pairs)
        cls._collected = True
        
    @classmethod
    def print_summary(cls):
        cls.collect()
        print("\n" + "="*60)
        print("PERFORMANCE PROFILE")
        print("="*60)
        print(f"{'Component':<30} | {'Total (s)':<10} | {'Calls':<8} | {'Avg (ms)':<10} | {'% Total':<8}")
        print("-" * 75)
        
        total_profiled = sum(cls._stats.values())
        if total_profiled == 0:
            print("No data collected.")
            return

        sorted_stats = sorted(cls._stats.items(), key=lambda x: x[1], reverse=True)
        
        for name, duration in sorted_stats:
            count = cls._counts[name]
            avg_ms = (duration / count) * 1000
            pct = (duration / total_profiled) * 100
            print(f"{name:<30} | {duration:<10.4f} | {count:<8} | {avg_ms:<10.3f} | {pct:<8.1f}")
        print("="*60 + "\n")

profiler = Profiler
