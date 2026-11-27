"""The Rust world agrees with an independent numpy implementation: signed distances to boxes, capsules and spheres,
terrain height (bilinear), and the collision verdict, on thousands of points around the obstacles and along the line.
Usage: python tests/test_world.py
"""
import json, os, sys
import numpy as np
import fpv_sim

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRACK = os.path.join(ROOT, "tracks", "hillside.json")
T = json.load(open(TRACK))
H = np.fromfile(os.path.join(ROOT, "tracks", T["terrain"]["file"]), "<f4").reshape(T["terrain"]["res"], -1).astype(float)


def rot(q, v):
    u = np.array(q[:3]); s = q[3]
    return 2 * (v @ u)[:, None] * u + (s * s - u @ u) * v + 2 * s * np.cross(u, v)


def distance(P):
    d = np.full(len(P), np.inf)
    for b in T["boxes"]:
        qi = [-b["q"][0], -b["q"][1], -b["q"][2], b["q"][3]]
        l = np.abs(rot(qi, P - b["c"])) - b["s"]
        d = np.minimum(d, np.linalg.norm(np.maximum(l, 0), axis=1) + np.minimum(l.max(1), 0))
    for k in T["capsules"]:
        a, b = np.array(k["a"]), np.array(k["b"]); ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(a + t[:, None] * ab - P, axis=1) - k["r"])
    for e in T["spheres"]:
        d = np.minimum(d, np.linalg.norm(P - e["c"], axis=1) - e["r"])
    return d


def height(x, z):
    o, s, n = T["terrain"]["origin"], T["terrain"]["size"], T["terrain"]["res"]
    fx = np.clip((x - o[0]) / (s[0] / (n - 1)), 0, n - 1.001); fz = np.clip((z - o[2]) / (s[2] / (n - 1)), 0, n - 1.001)
    i, j = np.floor(fz).astype(int), np.floor(fx).astype(int); a, b = fz - i, fx - j
    return H[i, j] * (1 - a) * (1 - b) + H[i, j + 1] * (1 - a) * b + H[i + 1, j] * a * (1 - b) + H[i + 1, j + 1] * a * b


E = fpv_sim.Racing(TRACK, 1)
rng = np.random.default_rng(1)
anchors = np.array([b["c"] for b in T["boxes"]] + [k["a"] for k in T["capsules"]] + [e["c"] for e in T["spheres"]])
P = np.ascontiguousarray(np.vstack([np.array(T["line"])[::8], anchors[rng.integers(0, len(anchors), 6000)] + rng.normal(0, 1.0, (6000, 3))]))
dr, hr, fr = E.probe(P)
dp = distance(P); hp = height(P[:, 0], P[:, 2])
Z = T["zone"]; inside = np.all((P >= Z["min"]) & (P <= Z["max"]), axis=1)
near = inside & (dp < T["drone_radius"] + 0.5)      # the grid keeps everything the drone can touch from a cell
free = inside & (P[:, 1] - T["drone_radius"] > hp)
e1 = np.abs(dr[near] - dp[near]).max(); e2 = int(((fr == 1) != (dp < T["drone_radius"]))[free].sum()); e3 = np.abs(hr - hp).max()
print(f"{len(P)} points, {near.sum()} within reach of an obstacle")
print(f"signed distance, max |d| {e1:.1e} m | collision verdict, disagreements {e2} of {free.sum()} | terrain, max |dh| {e3:.1e} m")
ok = e1 < 1e-9 and e2 == 0 and e3 < 1e-4
print("AGREE" if ok else "DISAGREE")
sys.exit(0 if ok else 1)
