"""The Rust base + residual step is the numpy reference step (python/residual.py), network included: same inputs, same
trajectory, to 1e-6 (in practice ~1e-12). Uses models/demo.json (run python/residual_demo.py first) and, if present,
any other model passed on the command line. Also checks that a race runs on the model.
Usage: python tests/test_residual.py [model.json]
"""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "python"))
import residual as R
import fpv_sim

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "demo.json")
m = R.load(path)
rng = np.random.default_rng(0)
e0 = np.array([0, 30, 0, 3, 0, 8, 0, 0, 0, 1, 0, 0, 0, 0.9])
worst = 0.0
for name, sticks in [("smooth flight", np.column_stack([-0.3 + 0.1 * np.sin(np.arange(2000) / 90), 0.2 * np.sin(np.arange(2000) / 150),
                                                        0.15 * np.sin(np.arange(2000) / 70), 0.15 * np.cos(np.arange(2000) / 110)])),
                     ("random stick steps", np.repeat(rng.uniform(-1, 1, (200, 4)), 10, axis=0))]:
    h = np.full(len(sticks), 0.25)                     # inside the ground-effect band if the model has one
    st = (e0[0:3].copy(), e0[3:6].copy(), e0[6:10].copy(), e0[10:13].copy(), e0[13]); prev = np.array([-0.4, 0, 0, 0])
    a = np.zeros((len(sticks), 14))
    for k, inp in enumerate(sticks):
        st = R.step(m, st, inp, prev, h[k]); prev = inp
        a[k] = [*st[0], *st[1], *st[2], *st[3], st[4]]
    b = fpv_sim.simulate_open_loop_model(e0, np.ascontiguousarray(sticks), np.array([-0.4, 0, 0, 0]), h, path)
    d = np.abs(a - b); worst = max(worst, d.max())
    print(f"{name}: {len(sticks)} steps | max |d| position {d[:, :3].max():.1e} m, velocity {d[:, 3:6].max():.1e} m/s, "
          f"quaternion {d[:, 6:10].max():.1e}")
# a race on the model: the environment loads it and runs
E = fpv_sim.Racing(os.path.join(ROOT, "tracks", "hillside.json"), 64, 1, {"model": path, "race": True, "res_min": 0.7, "res_max": 1.3, "bias_n": 0.2})
E.reset()
for _ in range(200):
    E.step(np.clip(rng.normal([-0.4, 0, 0, 0], 0.2, (64, 4)), -1, 1).astype(np.float32))
print(f"race on the model: {len(E.episodes()['end'])} episodes finished in 200 decisions of 64 random pilots")
ok = worst < 1e-6
print("EQUIVALENT" if ok else "DIFFERENT")
sys.exit(0 if ok else 1)
