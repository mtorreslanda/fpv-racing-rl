"""Throughput of the batched environment (random actions around hover: many crashes and resets, the worst case).
Usage: python tests/bench.py
"""
import os, time
import numpy as np
import fpv_sim

TRACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tracks", "hillside.json")
for n in [256, 1024, 4096, 16384]:
    E = fpv_sim.Racing(TRACK, n, 1)
    E.reset(); rng = np.random.default_rng(0)
    A = np.clip(rng.normal([-0.4, 0, 0, 0], 0.3, (32, n, 4)), -1, 1).astype(np.float32)
    E.step(A[0]); steps = 0; t = time.perf_counter()
    while time.perf_counter() - t < 3.0:
        for k in range(32):
            E.step(A[k]); steps += n
    dt = time.perf_counter() - t
    print(f"N={n:6d}: {steps/dt/1e6:5.2f} M decisions/s = {2*steps/dt/1e6:5.2f} M physics steps/s "
          f"(= {2*steps/dt*0.01/3600:5.1f} hours of flight per second)")
