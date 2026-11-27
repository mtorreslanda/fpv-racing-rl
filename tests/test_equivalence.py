"""The Rust physics step is the numpy reference step: same inputs, same trajectory, to 1e-6 (in practice ~1e-12).

Two runs: a smooth flight with gentle sticks, and 3,000 steps of random stick steps (flips included).
Usage: python tests/test_equivalence.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import fpv_sim_ref as S
import fpv_sim


def python(e0, sticks, prev0):
    st = (e0[0:3].copy(), e0[3:6].copy(), e0[6:10].copy(), e0[10:13].copy(), e0[13])
    out = np.zeros((len(sticks), 14)); prev = prev0; P = S.Params()
    for k, inp in enumerate(sticks):
        st = S.step(st, inp, prev, P); prev = inp
        out[k] = [*st[0], *st[1], *st[2], *st[3], st[4]]
    return out


def compare(name, e0, sticks, prev0):
    t = time.perf_counter(); a = python(e0, sticks, prev0); tp = time.perf_counter() - t
    t = time.perf_counter()
    b = fpv_sim.simulate_open_loop(np.ascontiguousarray(e0), np.ascontiguousarray(sticks), np.ascontiguousarray(prev0))
    tr = time.perf_counter() - t
    d = np.abs(a - b)
    print(f"{name}: {len(sticks)} steps | max |d| position {d[:, :3].max():.1e} m, velocity {d[:, 3:6].max():.1e} m/s, "
          f"quaternion {d[:, 6:10].max():.1e}, rate {d[:, 10:13].max():.1e} deg/s | flown {np.linalg.norm(a[-1, :3] - e0[:3]):.0f} m | "
          f"numpy {tp*1e3:.0f} ms, Rust {tr*1e3:.2f} ms")
    return d.max()


rng = np.random.default_rng(0)
e0 = np.array([0, 50, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.0])
smooth = np.column_stack([-0.35 + 0.1 * np.sin(np.arange(3000) / 90), 0.2 * np.sin(np.arange(3000) / 150),
                          0.15 * np.sin(np.arange(3000) / 70), 0.15 * np.cos(np.arange(3000) / 110)])
m1 = compare("smooth flight", e0, smooth, smooth[0])
steps = np.repeat(rng.uniform(-1, 1, (300, 4)), 10, axis=0)          # 0.1 s steps, flips included
m2 = compare("random stick steps", e0, steps, np.array([-1.0, 0, 0, 0]))
ok = max(m1, m2) < 1e-6
print("EQUIVALENT" if ok else "DIFFERENT")
sys.exit(0 if ok else 1)
