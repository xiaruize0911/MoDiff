import time
import torch
from collections import defaultdict
import threading

class Profiler:
    _stats = defaultdict(float)
    _counts = defaultdict(int)
    enabled = True
    
    @classmethod
    def reset(cls):
        cls._stats.clear()
        cls._counts.clear()
    
    @classmethod
    def start(cls, name):
        if not cls.enabled: return None
        torch.cuda.synchronize()
        return time.time()
        
    @classmethod
    def stop(cls, name, start_time):
        if not cls.enabled or start_time is None: return
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        cls._stats[name] += elapsed
        cls._counts[name] += 1
        
    @classmethod
    def print_summary(cls):
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
