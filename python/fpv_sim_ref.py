"""Reference implementation of the physics step in numpy (model v0.2).

This is the specification the Rust simulator is checked against (tests/test_equivalence.py, to 1e-9 over thousands
of steps). State: p (m), v (m/s), q (x, y, z, w), w (deg/s in body axes: x = pitch, y = yaw, z = roll), battery (0-1).
Input per step: sticks (thr, yaw, pitch, roll) in [-1, 1]. Integration: semi-implicit Euler, dt = 0.01 s.
"""
import numpy as np

DT = 0.01
G = 9.81
# stick -> deg/s, symmetric; identified from rate steps
STICK = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 1.0])
RATES = {
    "pitch": np.array([0.0, 8.8, 49.9, 121.3, 231.5, 424.4]),
    "roll":  np.array([0.0, 9.1, 53.1, 134.7, 276.5, 583.7]),
    "yaw":   np.array([0.0, 8.8, 51.7, 121.6, 221.3, 375.0]),
}
ALPHA = {"pitch": 0.66, "roll": 0.80, "yaw": 1.0}   # fraction of the rate error closed per step


class Params:
    """v0.2. Body axis: f = u^2 t0 + k_in u v_axis + k_ax v_axis|v_axis| + k_ob u v_plane; drag a v + b v|v| per axis."""
    def __init__(self, **kw):
        self.t0, self.k_in, self.k_ax, self.k_ob = 108.59, -1.720, -0.0154, -0.207
        self.fz = (0.0846, -0.0256)    # forward
        self.fx = (-0.0318, -0.0268)   # side
        self.__dict__.update(kw)


def target_rate(axis, s):
    return np.sign(s) * np.interp(abs(s), STICK, RATES[axis])


def qmul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2])


def rotate(q, v):
    u = q[:3]; s = q[3]
    return 2*np.dot(u, v)*u + (s*s - np.dot(u, u))*v + 2*s*np.cross(u, v)


def step(state, sticks, sticks_prev, P):
    p, v, q, w, bat = state
    thr = sticks[0]
    # rates: target with one step of delay, first order
    target = np.array([target_rate("pitch", sticks_prev[2]), target_rate("yaw", sticks_prev[1]),
                       target_rate("roll", sticks_prev[3])])
    alpha = np.array([ALPHA["pitch"], ALPHA["yaw"], ALPHA["roll"]])
    w = w + alpha * (target - w)
    # attitude: rotation in body axes
    th = np.radians(w) * DT; ang = np.linalg.norm(th)
    dq = np.array([*(th / ang * np.sin(ang / 2)), np.cos(ang / 2)]) if ang > 1e-12 else np.array([0, 0, 0, 1.0])
    q = qmul(q, dq); q /= np.linalg.norm(q)
    # forces
    up = rotate(q, np.array([0, 1.0, 0])); rt = rotate(q, np.array([1.0, 0, 0])); fw = rotate(q, np.array([0, 0, 1.0]))
    va, vr, vf = np.dot(v, up), np.dot(v, rt), np.dot(v, fw)
    u = (thr + 1) / 2; vpl = np.hypot(vr, vf)
    f_ax = u*u*P.t0 + P.k_in*u*va + P.k_ax*va*abs(va) + P.k_ob*u*vpl
    f_fw = P.fz[0]*vf + P.fz[1]*vf*abs(vf)
    f_rt = P.fx[0]*vr + P.fx[1]*vr*abs(vr)
    a = f_ax*up + f_fw*fw + f_rt*rt - np.array([0, G, 0])
    v = v + a*DT
    p = p + v*DT
    return (p, v, q, w, bat)
