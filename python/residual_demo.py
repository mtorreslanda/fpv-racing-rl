"""The base + residual pipeline, end to end, on synthetic flights.

A hidden "true" drone generates the flights. It is the identified v0.4 physics plus two things the base model cannot
express: motors that take time to spin up (thrust follows the throttle with a ~45 ms time constant) and a side force when yawing at
speed. The pipeline sees only the flight logs, exactly as it would see telemetry:

  1. fit_base       least squares for the thrust table and the aero coefficients
  2. train_residual the network on what the base leaves unexplained
  3. error_ahead    position error 1 s ahead on flights held out from both fits

Writes models/demo.json (a model the simulator can load) and media/residual.png.
Usage: python python/residual_demo.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import fpv_sim_ref as S   # noqa: E402
import residual as R      # noqa: E402

DT, G = 0.01, 9.81
V04 = dict(t0=108.59, k_in=-1.720, k_ax=-0.0154, k_ob=-0.207, fz=(-0.0869, -0.00989), fx=(-0.0232, -0.01943))


def true_step(st, inp, prev, lag):
    """The hidden drone: v0.4 + motor lag + a yaw-induced side force."""
    p, v, q, w, bat = st
    target = np.array([S.target_rate("pitch", prev[2]), S.target_rate("yaw", prev[1]), S.target_rate("roll", prev[3])])
    alpha = np.array([S.ALPHA["pitch"], S.ALPHA["yaw"], S.ALPHA["roll"]])
    w2 = w + alpha * (target - w)
    up = S.rotate(q, np.array([0, 1.0, 0])); rt = S.rotate(q, np.array([1.0, 0, 0])); fw = S.rotate(q, np.array([0, 0, 1.0]))
    va, vr, vf = v @ up, v @ rt, v @ fw
    u_cmd = (inp[0] + 1) / 2
    lag = lag + 0.22 * (u_cmd - lag)                                   # motors spin up with a ~45 ms time constant (0.01 s / 0.22)
    u = lag; s = np.linalg.norm(v)
    f_ax = u * u * V04["t0"] + V04["k_in"] * u * va + V04["k_ax"] * va * abs(va) + V04["k_ob"] * u * np.hypot(vr, vf)
    f_fw = V04["fz"][0] * vf + V04["fz"][1] * s * vf
    f_rt = V04["fx"][0] * vr + V04["fx"][1] * s * vr - 0.0009 * w[1] * vf     # side force when yawing at speed
    a = f_ax * up + f_fw * fw + f_rt * rt - np.array([0, G, 0])
    th = np.radians(w2) * DT; ang = np.linalg.norm(th)
    dq = np.array([*(th / ang * np.sin(ang / 2)), np.cos(ang / 2)]) if ang > 1e-12 else np.array([0, 0, 0, 1.0])
    q2 = S.qmul(q, dq); q2 /= np.linalg.norm(q2)
    v2 = v + a * DT
    return (p + v2 * DT, v2, q2, w2, bat), lag


def smooth_noise(rng, n, scale, corr):
    x = np.zeros(n); e = rng.normal(0, scale, n)
    for i in range(1, n): x[i] = corr * x[i - 1] + e[i]
    return x


def flight(rng, seconds=6.0):
    """A flight of random, smooth stick inputs around hover; throttle keeps the altitude roughly, the rates wander,
    and now and then a hard throttle punch or chop."""
    n = int(seconds / DT)
    st = (np.array([0, 40.0, 0]), rng.normal(0, 4, 3), np.array([0, 0, 0, 1.0]), np.zeros(3), 1.0); lag = 0.30
    yaw = smooth_noise(rng, n, .06, .98); pit = smooth_noise(rng, n, .05, .985); rol = smooth_noise(rng, n, .05, .985)
    punch = np.repeat(rng.choice([0, 0, 0, .35, -.25], n // 50 + 1), 50)[:n]
    P, V, Q, W, U = [], [], [], [], []
    prev = np.array([-.4, 0, 0, 0])
    for t in range(n):
        p, v, q, w, _ = st
        up = S.rotate(q, np.array([0, 1.0, 0]))
        thr = np.clip(-.4 + .08 * (40 - p[1]) - .06 * v[1] + punch[t], -1, 1)
        tilt = np.degrees(np.arccos(np.clip(up[1], -1, 1)))
        damp = -0.004 * w if tilt > 55 else 0 * w                         # keep it from flipping over
        inp = np.array([thr, np.clip(yaw[t], -1, 1), np.clip(pit[t] + damp[0], -1, 1), np.clip(rol[t] + damp[2], -1, 1)])
        P.append(p); V.append(v); Q.append(q); W.append(w); U.append(inp)
        st, lag = true_step(st, inp, prev, lag); prev = inp
    return dict(p=np.array(P), v=np.array(V), q=np.array(Q), w=np.array(W), sticks=np.array(U),
                bat=np.ones(n), h=np.full(n, 40.0))


def main():
    rng = np.random.default_rng(3)
    t0 = time.time()
    train = [flight(rng) for _ in range(160)]
    test = [flight(rng) for _ in range(40)]
    sp = np.concatenate([np.linalg.norm(f["v"], axis=1) for f in train])
    print(f"{len(train)} training flights, {len(test)} held out · speed up to {sp.max()*3.6:.0f} km/h · {time.time()-t0:.0f} s")
    m = R.fit_base(train)
    e_base = R.error_ahead(test, m)
    print(f"base only:       1 s ahead median {np.median(e_base):.2f} m, p90 {np.percentile(e_base, 90):.2f} m")
    m = R.train_residual(train, m)
    e_res = R.error_ahead(test, m)
    print(f"base + residual: 1 s ahead median {np.median(e_res):.2f} m, p90 {np.percentile(e_res, 90):.2f} m")
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    R.save(m, os.path.join(ROOT, "models", "demo.json"))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BG, INK, GREY, AMBER = "#0B0E0F", "#EAF2EE", "#7F8C87", "#FFB21F"
    plt.rcParams.update({"font.family": "monospace", "text.color": INK})
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=110); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bins = np.linspace(0, max(np.percentile(e_base, 98), .05), 40)
    ax.hist(e_base, bins, color="#3A4A50", label=f"base only · median {np.median(e_base):.2f} m")
    ax.hist(e_res, bins, color=AMBER, alpha=.85, label=f"base + residual · median {np.median(e_res):.2f} m")
    ax.set_xlabel("position error 1 s ahead, held-out flights (m)", color=GREY); ax.tick_params(colors=GREY)
    [s.set_color("#2A3034") for s in ax.spines.values()]; ax.legend(frameon=False, labelcolor=INK)
    fig.text(.06, .92, "THE RESIDUAL NETWORK · what the equations miss, learned from the flights", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=.85, bottom=.16)
    fig.savefig(os.path.join(ROOT, "media", "residual.png"), facecolor=BG); print("-> media/residual.png, models/demo.json")


if __name__ == "__main__":
    main()
