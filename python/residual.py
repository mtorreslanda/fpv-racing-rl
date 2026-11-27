"""The "base + residual" model: a grey-box physics base fitted by least squares, plus a small network that learns what
the equations miss. This is the structure of the model that let the policy transfer to the real game at racing speed;
here it is fitted from flight logs you provide (or from the synthetic demo in residual_demo.py).

Per step (100 Hz), in body axes [lateral, axial, forward]:
    thrust   F = u^2 * T(v_axial) * sag(battery) * ground(h)                    T: a table over axial speed
    aero_k   A_k = a v_k + b v_k|v_k| + c |V| v_k + d F v_k + e F                five coefficients per axis
    residual [dF, dA_lat, dA_axial, dA_fwd] = MLP(17 inputs)                     17 -> 64 -> 64 -> 4, tanh
    accel    = ((F + dF + A_axial + dA_axial) up + (A_lat + dA_lat) right + (A_fwd + dA_fwd) forward) / mass - g
Rotation is the identified rate model (physics.rs, v0.2). Everything is per unit mass by default (mass = 1).

The Rust simulator runs the same step (sim/src/residual.rs) from the same JSON file; tests/test_residual.py holds the two
together to 1e-9.

  fit_base(flights)                   least squares for T and the aero coefficients
  train_residual(flights, model)      the network, on what the base leaves unexplained
  error_ahead(flights, model, 1.0)    position error 1 s ahead, open loop, from many starting points
"""
import json
import numpy as np
import fpv_sim_ref as S

DT = 0.01
N_IN = 17


# ---------------------------------------------------------------------------------------------------------- the model
def empty_model(g=9.81, mass=1.0):
    """A model with a flat thrust table, no aero and no network: fill it with fit_base and train_residual."""
    return dict(mass=mass, g=g, t_x=[-40.0, 40.0], t_y=[0.0, 0.0], sag_x=[0.0, 0.03, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0],
                sag_y=[0.05, 0.05, 0.60, 0.70, 0.80, 0.92, 1.0, 1.0], ge_max=1.0, ge_height=0.0,
                aero=[[0.0] * 5 for _ in range(3)], net=None)


def save(model, path):
    json.dump(model, open(path, "w"), separators=(",", ":"))


def load(path):
    return json.load(open(path))


def ground(m, h):
    h = np.asarray(h, float)
    if m["ge_height"] <= 0:
        return np.ones_like(h)
    return np.where(h < m["ge_height"], np.maximum(m["ge_max"] + (1 - m["ge_max"]) * np.clip(h, 0, None) / m["ge_height"], 1.0), 1.0)


def thrust_base(m, u, v_ax, bat, h):
    return u ** 2 * np.interp(v_ax, m["t_x"], m["t_y"]) * np.interp(bat, m["sag_x"], m["sag_y"]) * ground(m, h)


def aero_features(vk, sp, F):
    return np.stack([vk, vk * np.abs(vk), sp * vk, F * vk, F], axis=-1)


def inputs(m, u, u1, sticks, sticks1, w, vb, sp, bat):
    """The network's 17 inputs: throttle now and one step before, yaw/pitch/roll sticks now and before, body rates,
    body velocity, speed, battery, and the thrust-table value at this axial speed."""
    tv = np.interp(vb[..., 1], m["t_x"], m["t_y"])
    return np.concatenate([u[..., None], u1[..., None], sticks, sticks1, w / 400.0, vb / 20.0, sp[..., None] / 20.0,
                           bat[..., None], (tv / 100.0)[..., None]], axis=-1)


def net(m, X):
    if not m.get("net"):
        return np.zeros(X.shape[:-1] + (4,))
    n = m["net"]
    x = (X - np.array(n["xm"])) / np.array(n["xs"])
    x = np.tanh(x @ np.array(n["w0"]).T + np.array(n["b0"]))
    x = np.tanh(x @ np.array(n["w1"]).T + np.array(n["b1"]))
    return (x @ np.array(n["w2"]).T + np.array(n["b2"])) * np.array(n["ys"])


def step(m, st, inp, prev, h=10.0):
    """One step of the base + residual model. st = (p, v, q, w [deg/s: pitch, yaw, roll], battery)."""
    p, v, q, w, bat = st
    qi = q * np.array([-1, -1, -1, 1.0])
    vb = S.rotate(qi, v); sp = np.linalg.norm(v); u = (inp[0] + 1) / 2; u1 = (prev[0] + 1) / 2
    X = inputs(m, np.array(u), np.array(u1), np.array([inp[1], inp[2], inp[3]]), np.array([prev[1], prev[2], prev[3]]),
               np.array(w), vb, np.array(sp), np.array(bat))
    r = net(m, X)
    # rotation: the identified rate model
    target = np.array([S.target_rate("pitch", prev[2]), S.target_rate("yaw", prev[1]), S.target_rate("roll", prev[3])])
    alpha = np.array([S.ALPHA["pitch"], S.ALPHA["yaw"], S.ALPHA["roll"]])
    w2 = w + alpha * (target - w)
    th = np.radians(w2) * DT; ang = np.linalg.norm(th)
    dq = np.array([*(th / ang * np.sin(ang / 2)), np.cos(ang / 2)]) if ang > 1e-12 else np.array([0, 0, 0, 1.0])
    q2 = S.qmul(q, dq); q2 /= np.linalg.norm(q2)
    # translation, with the attitude before the rotation step
    F = float(thrust_base(m, u, vb[1], bat, h)) + r[0]
    A = np.array([np.array(m["aero"][k]) @ aero_features(vb[k], sp, F) for k in range(3)]) + r[1:4]
    up = S.rotate(q, np.array([0, 1.0, 0])); rt = S.rotate(q, np.array([1.0, 0, 0])); fw = S.rotate(q, np.array([0, 0, 1.0]))
    Fw = (F + A[1]) * up + A[0] * rt + A[2] * fw
    v2 = v + (Fw / m["mass"] + np.array([0, -m["g"], 0])) * DT
    return (p + v2 * DT, v2, q2, w2, bat)


# ------------------------------------------------------------------------------------------------- fitting from flights
def body_quantities(f, m):
    """From a flight (dict of arrays at 100 Hz: p, v, q, w, sticks, bat, h), the body-frame velocity and the measured
    specific force along each body axis (what thrust + aero must explain), for every step t -> t+1."""
    v, q = f["v"], f["q"]
    acc = (v[1:] - v[:-1]) / DT + np.array([0, m["g"], 0])          # specific force in the world frame
    qi = q[:-1] * np.array([-1, -1, -1, 1.0])
    vb = np.array([S.rotate(a, b) for a, b in zip(qi, v[:-1])])
    fb = np.array([S.rotate(a, b) for a, b in zip(qi, acc)]) * m["mass"]
    u = (f["sticks"][:-1, 0] + 1) / 2
    return vb, fb, u


def fit_base(flights, m=None, bins=np.linspace(-30, 30, 25), iters=3):
    """Least squares, alternating: the thrust table T(v_axial) (piecewise linear, one value per bin) from the axial
    force with the current aero, then the aero coefficients of each axis with the current thrust."""
    m = m or empty_model()
    VB, FB, U, BAT, H = [], [], [], [], []
    for f in flights:
        vb, fb, u = body_quantities(f, m); VB.append(vb); FB.append(fb); U.append(u); BAT.append(f["bat"][:-1]); H.append(f["h"][:-1])
    vb, fb, u, bat, h = map(np.concatenate, (VB, FB, U, BAT, H))
    sp = np.linalg.norm(vb, axis=1)
    m["t_x"] = bins.tolist()
    scale = np.interp(bat, m["sag_x"], m["sag_y"]) * ground(m, h)
    # hat functions over the bins: T(v) = sum_j T_j phi_j(v)
    phi = np.stack([np.interp(vb[:, 1], bins, np.eye(len(bins))[j]) for j in range(len(bins))], axis=1)
    F = np.zeros(len(u))
    for _ in range(iters):
        aer_ax = aero_features(vb[:, 1], sp, F) @ np.array(m["aero"][1])
        Xt = (u ** 2 * scale)[:, None] * phi
        keep = u > 0.05
        Tj, *_ = np.linalg.lstsq(Xt[keep], (fb[:, 1] - aer_ax)[keep], rcond=None)
        m["t_y"] = Tj.tolist()
        F = thrust_base(m, u, vb[:, 1], bat, h)
        for k in range(3):
            target = fb[:, k] - (F if k == 1 else 0)
            c, *_ = np.linalg.lstsq(aero_features(vb[:, k], sp, F), target, rcond=None)
            m["aero"][k] = c.tolist()
    return m


def residual_targets(flights, m):
    """Inputs and targets for the network: [dF, dA_lat, dA_axial, dA_fwd] = what the base leaves of the measured force
    (dF and dA_axial are both along the body axis; the axial part is split evenly between them)."""
    X, Y = [], []
    for f in flights:
        vb, fb, u = body_quantities(f, m)
        sp = np.linalg.norm(vb, axis=1); bat = f["bat"][:-1]
        F = thrust_base(m, u, vb[:, 1], bat, f["h"][:-1])
        A = np.stack([aero_features(vb[:, k], sp, F) @ np.array(m["aero"][k]) for k in range(3)], axis=1)
        res = fb - A - np.stack([np.zeros_like(F), F, np.zeros_like(F)], axis=1)
        u1 = np.r_[u[0], u[:-1]]
        st = f["sticks"][:-1, 1:]; st1 = np.vstack([st[:1], st[:-1]])
        X.append(inputs(m, u, u1, st, st1, f["w"][:-1], vb, sp, bat))
        Y.append(np.stack([res[:, 1] / 2, res[:, 0], res[:, 1] / 2, res[:, 2]], axis=1))
    return np.concatenate(X), np.concatenate(Y)


def train_residual(flights, m, epochs=60, hidden=64, seed=0, lr=2e-3, wd=1e-5):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X, Y = residual_targets(flights, m)
    xm, xs = X.mean(0), X.std(0) + 1e-6; ys = Y.std(0) + 1e-6
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32); Yt = torch.tensor(Y / ys, dtype=torch.float32)
    netw = nn.Sequential(nn.Linear(N_IN, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 4))
    opt = torch.optim.Adam(netw.parameters(), lr=lr, weight_decay=wd)
    n = len(Xt); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]; loss = ((netw(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    L = [l for l in netw if isinstance(l, nn.Linear)]
    m["net"] = dict(w0=L[0].weight.tolist(), b0=L[0].bias.tolist(), w1=L[1].weight.tolist(), b1=L[1].bias.tolist(),
                    w2=L[2].weight.tolist(), b2=L[2].bias.tolist(), xm=xm.tolist(), xs=xs.tolist(), ys=ys.tolist())
    return m


def error_ahead(flights, m, seconds=1.0, every=50):
    """Open-loop position error after `seconds`, started from the true state every `every` steps of each flight."""
    n = int(seconds / DT); errs = []
    for f in flights:
        T = len(f["v"])
        for t0 in range(1, T - n - 1, every):
            st = (f["p"][t0].copy(), f["v"][t0].copy(), f["q"][t0].copy(), f["w"][t0].copy(), f["bat"][t0])
            for t in range(t0, t0 + n):
                st = step(m, st, f["sticks"][t], f["sticks"][t - 1], f["h"][t])
            errs.append(np.linalg.norm(st[0] - f["p"][t0 + n]))
    return np.array(errs)
