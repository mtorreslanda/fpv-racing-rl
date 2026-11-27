//! Quadcopter physics, identified from flight telemetry (no engine code, no game files).
//!
//! The model is a compact grey box: a first-order rate controller per body axis, thrust along the body axis that
//! falls off with the inflow, and quadratic drag in the body plane. Its parameters were fitted to recorded
//! flights and validated on whole trajectories the fit never saw (see the README).
//!
//! Conventions (left-handed, y up): x right, y up, z forward; quaternion (x, y, z, w); angular rate w in deg/s
//! in body axes (x = pitch, y = yaw, z = roll). Integration is semi-implicit Euler at 100 Hz: v += a·dt; p += v·dt.
//! `python/fpv_sim_ref.py` is the same step written in numpy; `tests/test_equivalence.py` holds the two to 1e-9.

pub const DT: f64 = 0.01;
pub const G: f64 = 9.81;
/// Stick deflection breakpoints of the rate curves.
pub const STICK: [f64; 6] = [0.0, 0.05, 0.25, 0.5, 0.75, 1.0];

/// Thrust factor against battery charge (0-1): flat while the pack is healthy, then sagging, and nearly nothing
/// when it is empty. A generic LiPo-like curve; the discharge rate below was fitted from flight telemetry.
pub const BAT_X: [f64; 7] = [0.0, 0.03, 0.10, 0.20, 0.30, 0.40, 0.50];
pub const BAT_F: [f64; 7] = [0.05, 0.05, 0.60, 0.70, 0.80, 0.92, 1.0];
/// Discharge (fraction per second) = max(0, D2·u² + D3·u³), u = throttle in [0, 1] (fitted on 15 flights, R² 0.93).
pub const DISCHARGE_U2: f64 = -0.0143;
pub const DISCHARGE_U3: f64 = 0.0516;

#[inline]
pub fn battery_factor(b: f64) -> f64 {
    let n = BAT_X.len();
    if b >= BAT_X[n - 1] {
        return 1.0;
    }
    if b <= 0.0 {
        return BAT_F[0];
    }
    let mut k = 0;
    while k < n - 2 && b >= BAT_X[k + 1] {
        k += 1;
    }
    BAT_F[k] + (BAT_F[k + 1] - BAT_F[k]) * (b - BAT_X[k]) / (BAT_X[k + 1] - BAT_X[k])
}

pub type V3 = [f64; 3];
pub type Q = [f64; 4];

#[derive(Clone, Debug)]
pub struct Params {
    // body axis: f = u²·t0·g(bat) + k_in·u·v_axis + k_ax·v_axis|v_axis| + k_ob·u·v_plane
    pub t0: f64,
    pub k_in: f64,
    pub k_ax: f64,
    pub k_ob: f64,
    /// forward drag (a, b)
    pub fz: [f64; 2],
    /// side drag (a, b)
    pub fx: [f64; 2],
    /// stick -> deg/s per body axis, in the order of w: [pitch, yaw, roll]
    pub rates: [[f64; 6]; 3],
    /// fraction of the rate error closed per step, [pitch, yaw, roll]
    pub alpha: [f64; 3],
    /// battery on: thrust scaled by battery_factor(bat), and the pack discharges
    pub battery: bool,
    /// discharge multiplier (1 = measured; randomised in training)
    pub discharge: f64,
    /// body-plane drag as a·v + b·|v|·v (the high-speed fit) instead of a·v + b·v|v| per axis
    pub drag_v04: bool,
}

impl Params {
    /// v0.2: fitted on hover, steps and laps up to 12 m/s; validated out of sample (1 s ahead: 4 cm on manoeuvres).
    pub fn v02() -> Self {
        Params {
            t0: 108.59,
            k_in: -1.720,
            k_ax: -0.0154,
            k_ob: -0.207,
            fz: [0.0846, -0.0256],
            fx: [-0.0318, -0.0268],
            rates: [
                [0.0, 8.8, 49.9, 121.3, 231.5, 424.4], // pitch
                [0.0, 8.8, 51.7, 121.6, 221.3, 375.0], // yaw
                [0.0, 9.1, 53.1, 134.7, 276.5, 583.7], // roll
            ],
            alpha: [0.66, 1.0, 0.80],
            battery: false,
            discharge: 1.0,
            drag_v04: false,
        }
    }

    /// v0.3 = v0.2 + battery (discharge and thrust sag).
    pub fn v03() -> Self {
        Params { battery: true, ..Params::v02() }
    }

    /// v0.4 = v0.3 + drag a·v + b·|v|·v in the body plane, refitted on fast straights and slaloms up to 46 m/s
    /// (out of sample it improved every validation: 1 s ahead on a 3-lap race, 0.51 -> 0.28 m).
    pub fn v04() -> Self {
        Params { drag_v04: true, fz: [-0.0869, -0.00989], fx: [-0.0232, -0.01943], ..Params::v03() }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    pub p: V3,
    pub v: V3,
    pub q: Q,
    pub w: V3,
    pub bat: f64,
}

#[inline]
pub fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
pub fn norm(a: &V3) -> f64 {
    dot(a, a).sqrt()
}

/// np.interp(|s|, STICK, table) with the sign of s (saturates outside [0, 1]).
#[inline]
pub fn target_rate(table: &[f64; 6], s: f64) -> f64 {
    let x = s.abs();
    let y = if x >= STICK[5] {
        table[5]
    } else {
        let mut k = 0;
        while k < 4 && x >= STICK[k + 1] {
            k += 1;
        }
        // same form as np.interp: slope·(x − xp[k]) + fp[k]
        let slope = (table[k + 1] - table[k]) / (STICK[k + 1] - STICK[k]);
        slope * (x - STICK[k]) + table[k]
    };
    if s > 0.0 {
        y
    } else if s < 0.0 {
        -y
    } else {
        0.0
    }
}

#[inline]
pub fn qmul(a: &Q, b: &Q) -> Q {
    let (x1, y1, z1, w1) = (a[0], a[1], a[2], a[3]);
    let (x2, y2, z2, w2) = (b[0], b[1], b[2], b[3]);
    [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
}

/// Rotates v by q: 2(u·v)u + (s² − u·u)v + 2s(u × v).
#[inline]
pub fn rotate(q: &Q, v: &V3) -> V3 {
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    let a = 2.0 * dot(&u, v);
    let b = s * s - dot(&u, &u);
    let c = cross(&u, v);
    [
        a * u[0] + b * v[0] + 2.0 * s * c[0],
        a * u[1] + b * v[1] + 2.0 * s * c[1],
        a * u[2] + b * v[2] + 2.0 * s * c[2],
    ]
}

/// One physics step. `inp` = sticks (thr, yaw, pitch, roll) applied at this step; `inp_prev` = the previous
/// step's (rotation responds one step later than thrust, as measured).
#[inline]
pub fn step(st: &mut State, inp: &[f64; 4], inp_prev: &[f64; 4], p: &Params) {
    // rates: target with one step of delay, first order
    let target = [
        target_rate(&p.rates[0], inp_prev[2]),
        target_rate(&p.rates[1], inp_prev[1]),
        target_rate(&p.rates[2], inp_prev[3]),
    ];
    for i in 0..3 {
        st.w[i] = st.w[i] + p.alpha[i] * (target[i] - st.w[i]);
    }
    // attitude
    let rad = std::f64::consts::PI / 180.0;
    let th = [st.w[0] * rad * DT, st.w[1] * rad * DT, st.w[2] * rad * DT];
    let ang = norm(&th);
    let dq = if ang > 1e-12 {
        let s = (ang / 2.0).sin();
        [th[0] / ang * s, th[1] / ang * s, th[2] / ang * s, (ang / 2.0).cos()]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let mut q = qmul(&st.q, &dq);
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    for k in 0..4 {
        q[k] /= n;
    }
    st.q = q;
    // forces
    let up = rotate(&q, &[0.0, 1.0, 0.0]);
    let rt = rotate(&q, &[1.0, 0.0, 0.0]);
    let fw = rotate(&q, &[0.0, 0.0, 1.0]);
    let (va, vr, vf) = (dot(&st.v, &up), dot(&st.v, &rt), dot(&st.v, &fw));
    let u = (inp[0] + 1.0) / 2.0;
    let vpl = vr.hypot(vf);
    let g = if p.battery { battery_factor(st.bat) } else { 1.0 };
    let f_ax = u * u * p.t0 * g + p.k_in * u * va + p.k_ax * va * va.abs() + p.k_ob * u * vpl;
    let (f_fw, f_rt) = if p.drag_v04 {
        let s = norm(&st.v);
        (p.fz[0] * vf + p.fz[1] * s * vf, p.fx[0] * vr + p.fx[1] * s * vr)
    } else {
        (p.fz[0] * vf + p.fz[1] * vf * vf.abs(), p.fx[0] * vr + p.fx[1] * vr * vr.abs())
    };
    for i in 0..3 {
        let a = f_ax * up[i] + f_fw * fw[i] + f_rt * rt[i] - if i == 1 { G } else { 0.0 };
        st.v[i] = st.v[i] + a * DT;
    }
    for i in 0..3 {
        st.p[i] = st.p[i] + st.v[i] * DT;
    }
    if p.battery {
        let d = (DISCHARGE_U2 * u * u + DISCHARGE_U3 * u * u * u).max(0.0);
        st.bat = (st.bat - p.discharge * d * DT).max(0.0);
    }
}
