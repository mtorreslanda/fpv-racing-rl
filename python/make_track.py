"""Builds the synthetic race track `tracks/hillside.json` (+ its height map `hillside_terrain.bin`).

Everything here is invented: a 360 x 300 m field of rolling hills, 12 gates on a closed loop with a climb, a dive off
the ridge, a hairpin and a slalom, gate frames that can be hit, poles, round bales and trees. The racing line is a
smooth closed spline through the gate centres, kept >= 1.2 m over the ground.

Usage: python python/make_track.py            (deterministic: same file every time)
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tracks")
RES, SIZE, ORIGIN = 257, 400.0, (-200.0, 0.0, -200.0)
RADIUS = 0.12            # drone collision sphere (m)
HW, HH = 1.5, 1.2        # gate opening: 3.0 x 2.4 m
BAR = 0.12               # half thickness of the gate frame


def terrain_height(x, z):
    """Rolling hills: a long ridge across the north, two knolls and a gentle valley."""
    h = 9.0 * np.exp(-(((z - 70) / 34) ** 2)) * (0.75 + 0.25 * np.cos(x / 60))
    h += 5.0 * np.exp(-(((x + 95) ** 2 + (z + 40) ** 2) / (2 * 26 ** 2)))
    h += 4.0 * np.exp(-(((x - 110) ** 2 + (z + 70) ** 2) / (2 * 30 ** 2)))
    h += 0.6 * np.sin(x / 17.0) * np.cos(z / 23.0)
    return h


# gates in race order: (x, z, height of the centre above the ground, name)
GATES = [
    (0.0, -95.0, 1.8, "start / finish"),
    (55.0, -85.0, 1.8, "sprint"),
    (105.0, -55.0, 2.2, "knoll"),
    (125.0, 5.0, 1.8, "turn east"),
    (95.0, 55.0, 2.4, "ridge climb"),
    (45.0, 78.0, 2.0, "ridge top"),
    (-10.0, 60.0, 1.8, "dive"),
    (-55.0, 40.0, 1.6, "valley"),
    (-110.0, 30.0, 1.8, "hairpin in"),
    (-130.0, -15.0, 1.8, "hairpin out"),
    (-90.0, -60.0, 2.0, "slalom"),
    (-45.0, -95.0, 1.8, "home straight"),
]


def catmull_rom_closed(P, n_per_seg=400):
    P = np.asarray(P, float); N = len(P); out = []
    for i in range(N):
        p0, p1, p2, p3 = P[(i - 1) % N], P[i], P[(i + 1) % N], P[(i + 2) % N]
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def resample(L, ds=0.25):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
    t = np.arange(0, s[-1], ds)
    return np.column_stack([np.interp(t, s, L[:, k]) for k in range(3)])


def quat_y(yaw):
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


def main():
    os.makedirs(OUT, exist_ok=True)
    xs = ORIGIN[0] + np.arange(RES) * SIZE / (RES - 1)
    zs = ORIGIN[2] + np.arange(RES) * SIZE / (RES - 1)
    X, Z = np.meshgrid(xs, zs)                       # row = z, column = x
    H = terrain_height(X, Z).astype("<f4")
    H.tofile(os.path.join(OUT, "hillside_terrain.bin"))

    C = np.array([[x, terrain_height(x, z) + h, z] for x, z, h, _ in GATES])
    line = resample(catmull_rom_closed(C))
    ground = terrain_height(line[:, 0], line[:, 2])
    line[:, 1] = np.maximum(line[:, 1], ground + 1.2)

    gates, boxes, capsules, spheres = [], [], [], []
    for k, (x, z, h, name) in enumerate(GATES):
        i = int(np.argmin(np.linalg.norm(line - C[k], axis=1)))
        t = line[(i + 4) % len(line)] - line[i - 4]; t[1] = 0; t /= np.linalg.norm(t)
        right = np.array([t[2], 0.0, -t[0]])
        c = C[k]
        gates.append(dict(id=k, name=name, center=c.round(4).tolist(), right=right.round(6).tolist(), up=[0.0, 1.0, 0.0],
                          forward=t.round(6).tolist(), half_width=HW, half_height=HH))
        q = quat_y(math.atan2(t[0], t[2]))
        g = terrain_height(x, z)
        # frame: two posts down to the ground and a top bar (the opening itself is free)
        for side in (-1, 1):
            pc = c + side * (HW + BAR) * right
            top = c[1] + HH + 2 * BAR
            boxes.append(dict(c=[pc[0], (g + top) / 2, pc[2]], s=[BAR, (top - g) / 2, BAR], q=q))
        boxes.append(dict(c=[c[0], c[1] + HH + BAR, c[2]], s=[HW + 2 * BAR, BAR, BAR], q=q))

    # props, kept >= 3 m from the racing line (they shape the space, they do not block it)
    rng = np.random.default_rng(7)
    def free(p, d=3.0):
        return np.min(np.linalg.norm(line[:, [0, 2]] - p, axis=1)) > d
    # slalom poles between gates 9 and 11, alternating sides of the line
    seg = line[int(np.argmin(np.linalg.norm(line - C[9], axis=1))):int(np.argmin(np.linalg.norm(line - C[11], axis=1)))]
    for j, idx in enumerate(np.linspace(40, len(seg) - 40, 7).astype(int)):
        p = seg[idx]; t = seg[min(idx + 4, len(seg) - 1)] - seg[idx - 4]; t[1] = 0; t /= np.linalg.norm(t)
        n = np.array([t[2], 0, -t[0]]) * (2.4 if j % 2 else -2.4)
        b = p + n; gb = terrain_height(b[0], b[2])
        capsules.append(dict(a=[b[0], gb, b[2]], b=[b[0], gb + 4.0, b[2]], r=0.15))
    # round bales lying on the ground, and trees, scattered where the line is not
    n_bales = n_trees = 0
    while n_bales < 40:
        p = rng.uniform([-170, -170], [170, 140])
        if not free(p, 4.0): continue
        yaw = rng.uniform(0, math.pi); d = np.array([math.cos(yaw), 0, math.sin(yaw)]) * 0.7
        g = terrain_height(p[0], p[1]) + 0.75
        capsules.append(dict(a=[p[0] - d[0], g, p[1] - d[2]], b=[p[0] + d[0], g, p[1] + d[2]], r=0.75)); n_bales += 1
    while n_trees < 45:
        p = rng.uniform([-185, -185], [185, 160])
        if not free(p, 6.0): continue
        g = terrain_height(p[0], p[1]); hgt = rng.uniform(5, 9)
        capsules.append(dict(a=[p[0], g, p[1]], b=[p[0], g + hgt, p[1]], r=0.25))
        spheres.append(dict(c=[p[0], g + hgt, p[1]], r=rng.uniform(1.8, 3.0))); n_trees += 1

    start = C[0] - 12.0 * np.array(gates[0]["forward"]); start[1] = terrain_height(start[0], start[2]) + 0.15
    track = dict(
        name="hillside", note="Synthetic track for fpv-racing-rl (python/make_track.py).",
        terrain=dict(file="hillside_terrain.bin", origin=list(ORIGIN), size=[SIZE, 80.0, SIZE], res=RES),
        zone=dict(min=[-195.0, -5.0, -195.0], max=[195.0, 60.0, 175.0]),
        start=start.round(4).tolist(), drone_radius=RADIUS, gates=gates, boxes=boxes, capsules=capsules,
        spheres=spheres, line=line.round(4).tolist())
    json.dump(track, open(os.path.join(OUT, "hillside.json"), "w"))
    L = np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))
    print(f"hillside: {len(gates)} gates, lap {L:.0f} m, {len(boxes)} boxes, {len(capsules)} capsules, {len(spheres)} spheres, "
          f"climb {line[:, 1].min():.1f} -> {line[:, 1].max():.1f} m")


if __name__ == "__main__":
    main()
