//! The "base + residual" model (python/residual.py is the reference; tests/test_residual.py holds the two together).
//! Base, in body axes [lateral, axial, forward]: thrust u²·T(v_axial)·sag(battery)·ground(h) and five aero terms per
//! axis [v, v|v|, |V|·v, F·v, F]; residual: an MLP 17 -> H -> H -> 4 that corrects [thrust, aero lateral, aero axial,
//! aero forward]. Rotation is the identified rate model of physics.rs. Loaded from the JSON a fit writes.

use crate::physics::{norm, qmul, rotate, target_rate, Params, State, DISCHARGE_U2, DISCHARGE_U3, DT};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Net {
    pub w0: Vec<Vec<f64>>,
    pub b0: Vec<f64>,
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
    pub xm: Vec<f64>,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Model {
    pub mass: f64,
    pub g: f64,
    pub t_x: Vec<f64>,
    pub t_y: Vec<f64>,
    pub sag_x: Vec<f64>,
    pub sag_y: Vec<f64>,
    pub ge_max: f64,
    pub ge_height: f64,
    pub aero: Vec<Vec<f64>>,
    pub net: Option<Net>,
}

/// np.interp (clamped at the ends).
#[inline]
pub fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let slope = (ys[hi] - ys[lo]) / (xs[hi] - xs[lo]);
    slope * (x - xs[lo]) + ys[lo]
}

impl Model {
    pub fn load(path: &str) -> Result<Model, String> {
        let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let m: Model = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        if m.aero.len() != 3 || m.aero.iter().any(|a| a.len() != 5) || m.t_x.len() != m.t_y.len() || m.t_x.len() < 2 {
            return Err("model: unexpected dimensions (aero 3x5, thrust table x/y of equal length)".into());
        }
        if let Some(n) = &m.net {
            if n.w0.is_empty() || n.w0[0].len() != 17 || n.w2.len() != 4 || n.b0.len() > 128 || n.b1.len() > 128 {
                return Err("model: the network must be 17 -> (<=128) -> (<=128) -> 4".into());
            }
        }
        Ok(m)
    }

    pub fn ground(&self, h: f64) -> f64 {
        if self.ge_height > 0.0 && h < self.ge_height {
            (self.ge_max + (1.0 - self.ge_max) * h.max(0.0) / self.ge_height).max(1.0)
        } else {
            1.0
        }
    }

    /// The residual network on its 17 raw inputs; zeros when the model has no network.
    pub fn net(&self, x: &[f64; 17]) -> [f64; 4] {
        let n = match &self.net {
            Some(n) => n,
            None => return [0.0; 4],
        };
        let mut a = [0.0f64; 17];
        for i in 0..17 {
            a[i] = (x[i] - n.xm[i]) / n.xs[i];
        }
        let (n1, n2) = (n.b0.len(), n.b1.len());
        let mut h1 = [0.0f64; 128];
        for j in 0..n1 {
            let mut s = n.b0[j];
            for i in 0..17 {
                s += n.w0[j][i] * a[i];
            }
            h1[j] = s.tanh();
        }
        let mut h2 = [0.0f64; 128];
        for j in 0..n2 {
            let mut s = n.b1[j];
            for i in 0..n1 {
                s += n.w1[j][i] * h1[i];
            }
            h2[j] = s.tanh();
        }
        let mut out = [0.0f64; 4];
        for k in 0..4 {
            let mut s = n.b2[k];
            for i in 0..n2 {
                s += n.w2[k][i] * h2[i];
            }
            out[k] = s * n.ys[k];
        }
        out
    }
}

/// One step with the base + residual model. `h` = height over the terrain (ground effect). `k_thrust` and `k_aero`
/// scale the base (randomisation; 1 = nominal), `k_res` scales the network's correction and `bias` is a constant
/// body-frame force [lateral, axial, forward]: both train a policy for the uncertainty the model still has.
pub fn step_residual(st: &mut State, inp: &[f64; 4], inp_prev: &[f64; 4], p: &Params, m: &Model, h: f64, k_thrust: f64, k_aero: f64, k_res: f64, bias: &[f64; 3]) {
    let q_old = st.q;
    let v = st.v;
    let qi = [-q_old[0], -q_old[1], -q_old[2], q_old[3]];
    let vb = rotate(&qi, &v);
    let sp = norm(&v);
    let u = (inp[0] + 1.0) / 2.0;
    let u1 = (inp_prev[0] + 1.0) / 2.0;
    let tv = interp(&m.t_x, &m.t_y, vb[1]);
    let x = [
        u, u1, inp[1], inp[2], inp[3], inp_prev[1], inp_prev[2], inp_prev[3],
        st.w[0] / 400.0, st.w[1] / 400.0, st.w[2] / 400.0,
        vb[0] / 20.0, vb[1] / 20.0, vb[2] / 20.0, sp / 20.0, st.bat, tv / 100.0,
    ];
    let mut r = m.net(&x);
    for k in 0..4 {
        r[k] *= k_res;
    }
    // rotation: the identified rate model
    let target = [target_rate(&p.rates[0], inp_prev[2]), target_rate(&p.rates[1], inp_prev[1]), target_rate(&p.rates[2], inp_prev[3])];
    let mut w = st.w;
    for i in 0..3 {
        w[i] = w[i] + p.alpha[i] * (target[i] - w[i]);
    }
    let rad = std::f64::consts::PI / 180.0;
    let th = [w[0] * rad * DT, w[1] * rad * DT, w[2] * rad * DT];
    let ang = norm(&th);
    let dq = if ang > 1e-12 {
        let s = (ang / 2.0).sin();
        [th[0] / ang * s, th[1] / ang * s, th[2] / ang * s, (ang / 2.0).cos()]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let mut q2 = qmul(&q_old, &dq);
    let nq = (q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2] + q2[3] * q2[3]).sqrt();
    for k in 0..4 {
        q2[k] /= nq;
    }
    // translation, with the attitude before the rotation step
    let f = u * u * tv * interp(&m.sag_x, &m.sag_y, st.bat) * m.ground(h) * k_thrust + r[0];
    let mut a = [0.0f64; 3];
    for k in 0..3 {
        let vk = vb[k];
        let c = &m.aero[k];
        a[k] = (c[0] * vk + c[1] * vk * vk.abs() + c[2] * sp * vk + c[3] * f * vk) * k_aero + c[4] * f + r[1 + k] + bias[k];
    }
    let up = rotate(&q_old, &[0.0, 1.0, 0.0]);
    let rt = rotate(&q_old, &[1.0, 0.0, 0.0]);
    let fw = rotate(&q_old, &[0.0, 0.0, 1.0]);
    for i in 0..3 {
        let fi = (f + a[1]) * up[i] + a[0] * rt[i] + a[2] * fw[i];
        st.v[i] = v[i] + (fi / m.mass - if i == 1 { m.g } else { 0.0 }) * DT;
        st.p[i] = st.p[i] + st.v[i] * DT;
    }
    st.q = q2;
    st.w = w;
    if p.battery {
        let d = (DISCHARGE_U2 * u * u + DISCHARGE_U3 * u * u * u).max(0.0);
        st.bat = (st.bat - p.discharge * d * DT).max(0.0);
    }
}
