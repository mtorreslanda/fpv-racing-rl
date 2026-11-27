//! Racing environment for RL: N independent copies, automatic reset.
//! The policy acts every `decim` physics steps (50 Hz with decim = 2); its command reaches the physics after
//! `delay` steps (thrust) and delay + 1 (rotation), as measured on the real controller-to-telemetry loop.

use crate::physics::{dot, norm, qmul, rotate, step, Params, State, DT, V3};
use crate::world::{End, World};
use std::sync::Arc;

pub const GATES_IN_OBS: usize = 3;
pub const OBS: usize = 23 + 12 * GATES_IN_OBS; // 59: body velocity, attitude, rates, last action, 3 gates, height, battery, laps left, gate index
pub const RACE_LAPS: u32 = 3;
const HIST: usize = 8;
pub const THR_HOVER: f64 = -0.399; // u² · 108.59 = 9.81

#[derive(Clone, Debug)]
pub struct Cfg {
    pub decim: usize,
    pub max_steps: usize,
    pub randomize: bool,
    pub delay: usize,
    pub k_progress: f64,
    pub k_gate: f64,
    pub k_rate: f64,
    pub k_smooth: f64,
    pub k_crash: f64,
    pub v_init_max: f64,
    pub p_start: f64,
    /// each gate opening shrinks by this much per side (be stricter than the rule)
    pub gate_margin: f64,
    /// per-step penalty k_near·(1 − clearance/d_near)² when clearance < d_near
    pub k_near: f64,
    pub d_near: f64,
    /// battery at the start: U(bat_min, 1), or bat_fixed if > 0
    pub bat_min: f64,
    pub bat_fixed: f64,
    /// race mode: the episode ends (well) after the target laps; from the start pad = 3 laps on a full pack,
    /// from a random point of the line = 1-3 laps with a random charge
    pub race: bool,
    /// reward for finishing the race
    pub k_finish: f64,
    /// safety margin (m): getting closer than this to terrain or an obstacle counts as a crash
    pub margin: f64,
    /// per-decision time penalty (0 = none)
    pub k_time: f64,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            decim: 2,
            max_steps: 1500,
            randomize: true,
            delay: 2,
            k_progress: 1.0,
            k_gate: 2.0,
            k_rate: 2e-4,
            k_smooth: 1e-4,
            k_crash: 5.0,
            v_init_max: 10.0,
            p_start: 0.1,
            gate_margin: 0.3,
            k_near: 0.1,
            d_near: 0.4,
            bat_min: 0.4,
            bat_fixed: 0.0,
            race: false,
            k_finish: 20.0,
            margin: 0.0,
            k_time: 0.0,
        }
    }
}

/// Summary of a finished episode.
#[derive(Clone, Copy, Debug)]
pub struct Episode {
    pub ret: f64,
    pub steps: usize,
    pub gates: u32,
    pub laps: u32,
    pub best_lap: f64,
    /// time from the start of the episode to RACE_LAPS laps (NaN if not reached)
    pub race_time: f64,
    pub bat_end: f64,
    pub end: End,
}

pub struct Env {
    pub st: State,
    pub par: Params,
    hist: [[f64; 4]; HIST],
    t: usize,
    pub delay: usize,
    pub gate: usize,
    a_prev: [f64; 4],
    pub steps: usize,
    pub time: f64,
    d_prev: f64,
    rng: u64,
    t_finish_line: Option<f64>,
    pub laps: u32,
    pub last_lap: f64,
    pub best_lap: f64,
    pub race_time: f64,
    /// laps to complete in this episode (after the first pass through the finish gate)
    pub laps_target: u32,
    pub gates: u32,
    ret: f64,
    pub end: End,
    pub finished: Vec<Episode>,
}

#[inline]
fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
#[inline]
fn unif(s: &mut u64, a: f64, b: f64) -> f64 {
    a + (b - a) * ((splitmix(s) >> 11) as f64 / (1u64 << 53) as f64)
}

fn q_axis(axis: &V3, ang: f64) -> [f64; 4] {
    let s = (ang / 2.0).sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, (ang / 2.0).cos()]
}

impl Env {
    pub fn new(seed: u64) -> Env {
        Env {
            st: State::default(),
            par: Params::v02(),
            hist: [[THR_HOVER, 0.0, 0.0, 0.0]; HIST],
            t: HIST,
            delay: 2,
            gate: 0,
            a_prev: [THR_HOVER, 0.0, 0.0, 0.0],
            steps: 0,
            time: 0.0,
            d_prev: 0.0,
            rng: seed,
            t_finish_line: None,
            laps: 0,
            last_lap: f64::NAN,
            best_lap: f64::NAN,
            race_time: f64::NAN,
            laps_target: RACE_LAPS,
            gates: 0,
            ret: 0.0,
            end: End::None,
            finished: Vec::new(),
        }
    }

    /// Sets the state by hand (tests and external loops): command history filled with `cmd`.
    pub fn set(&mut self, m: &World, st: State, cmd: [f64; 4], gate: usize, delay: usize) {
        self.st = st;
        self.hist = [cmd; HIST];
        self.t = HIST;
        self.delay = delay;
        self.gate = gate;
        self.a_prev = cmd;
        self.steps = 0;
        self.time = 0.0;
        self.t_finish_line = None;
        self.laps = 0;
        self.last_lap = f64::NAN;
        self.best_lap = f64::NAN;
        self.race_time = f64::NAN;
        self.laps_target = RACE_LAPS;
        self.gates = 0;
        self.ret = 0.0;
        self.end = End::None;
        self.d_prev = self.gate_distance(m);
    }

    fn gate_distance(&self, m: &World) -> f64 {
        let c = m.gates[self.gate].c;
        norm(&[self.st.p[0] - c[0], self.st.p[1] - c[1], self.st.p[2] - c[2]])
    }

    pub fn reset(&mut self, m: &World, cfg: &Cfg) {
        let s = &mut self.rng;
        // physics of this episode (domain randomisation)
        let mut par = Params::v04();
        let mut delay = cfg.delay;
        if cfg.randomize {
            par.t0 *= unif(s, 0.95, 1.05);
            let k = unif(s, 0.8, 1.2);
            par.fz = [par.fz[0] * k, par.fz[1] * k];
            par.fx = [par.fx[0] * k, par.fx[1] * k];
            par.k_ax *= k;
            let kr = unif(s, 0.97, 1.03);
            for e in par.rates.iter_mut() {
                for x in e.iter_mut() {
                    *x *= kr;
                }
            }
            delay = 1 + (splitmix(s) % 3) as usize;
            par.discharge *= unif(s, 0.85, 1.15);
        }
        let mut bat0 = if cfg.bat_fixed > 0.0 { cfg.bat_fixed } else { unif(s, cfg.bat_min, 1.0) };
        let mut target = RACE_LAPS;
        // start: above the pad, or at a random point of the reference line
        let mut st = State { bat: bat0, q: [0.0, 0.0, 0.0, 1.0], ..Default::default() };
        let mut gate = 0;
        for _attempt in 0..20 {
            let (p, tan, g, on_pad) = if unif(s, 0.0, 1.0) < cfg.p_start {
                let p = [m.start[0], m.start[1] + unif(s, 1.0, 5.0), m.start[2]];
                let c = m.gates[0].c;
                let d = [c[0] - p[0], 0.0, c[2] - p[2]];
                let l = norm(&d).max(1e-9);
                (p, [d[0] / l, 0.0, d[2] / l], 0usize, true)
            } else {
                let i = (splitmix(s) % m.line.len() as u64) as usize;
                let r = m.line[i];
                let p = [r[0] + unif(s, -0.5, 0.5), r[1] + unif(s, -0.5, 0.5), r[2] + unif(s, -0.5, 0.5)];
                (p, m.line_tan[i], m.line_gate[i], false)
            };
            let v0 = if on_pad { 0.0 } else { unif(s, 0.0, cfg.v_init_max) };
            let heading = tan[0].atan2(tan[2]);
            let q_h = q_axis(&[0.0, 1.0, 0.0], heading);
            let q_p = q_axis(&[1.0, 0.0, 0.0], unif(s, -0.25, 0.25));
            let q_r = q_axis(&[0.0, 0.0, 1.0], unif(s, -0.25, 0.25));
            st.p = p;
            st.v = [tan[0] * v0, tan[1] * v0, tan[2] * v0];
            st.q = qmul(&qmul(&q_h, &q_p), &q_r);
            st.w = [0.0; 3];
            gate = g;
            if cfg.race {
                if on_pad {
                    target = RACE_LAPS;
                    if cfg.bat_fixed <= 0.0 {
                        bat0 = 1.0;
                    }
                } else {
                    target = 1 + (splitmix(s) % RACE_LAPS as u64) as u32;
                }
                st.bat = bat0;
            }
            if m.collision(&p) == End::None {
                break;
            }
        }
        self.par = par;
        self.set(m, st, [THR_HOVER, 0.0, 0.0, 0.0], gate, delay);
        self.laps_target = target;
    }

    /// One policy decision. Returns (reward, terminated, truncated); does not reset.
    pub fn step(&mut self, m: &World, cfg: &Cfg, a: [f64; 4]) -> (f64, bool, bool) {
        let a = [a[0].clamp(-1.0, 1.0), a[1].clamp(-1.0, 1.0), a[2].clamp(-1.0, 1.0), a[3].clamp(-1.0, 1.0)];
        let mut r = 0.0;
        for _ in 0..cfg.decim {
            self.hist[self.t % HIST] = a;
            let inp = self.hist[(self.t - self.delay) % HIST];
            let prev = self.hist[(self.t - self.delay - 1) % HIST];
            let p0 = self.st.p;
            step(&mut self.st, &inp, &prev, &self.par);
            self.t += 1;
            self.time += DT;
            if m.crosses(self.gate, &p0, &self.st.p, cfg.gate_margin) {
                self.gates += 1;
                r += cfg.k_gate;
                if self.gate == 0 {
                    if let Some(t0) = self.t_finish_line {
                        let v = self.time - t0;
                        self.laps += 1;
                        self.last_lap = v;
                        self.best_lap = if self.best_lap.is_nan() { v } else { self.best_lap.min(v) };
                        if self.laps == RACE_LAPS {
                            self.race_time = self.time;
                        }
                    }
                    self.t_finish_line = Some(self.time);
                }
                // progress towards the old gate up to the crossing; the next one starts from here
                let c = m.gates[self.gate].c;
                let d = norm(&[self.st.p[0] - c[0], self.st.p[1] - c[1], self.st.p[2] - c[2]]);
                r += cfg.k_progress * (self.d_prev - d);
                self.gate = (self.gate + 1) % m.gates.len();
                self.d_prev = self.gate_distance(m);
            }
            if cfg.race && self.laps >= self.laps_target {
                self.end = End::Finish;
                r += cfg.k_finish;
                break;
            }
            self.end = m.collision(&self.st.p);
            if self.end == End::None && cfg.margin > 0.0 && m.clearance(&self.st.p) < cfg.margin {
                self.end = End::Margin;
            }
            if self.end != End::None {
                break;
            }
        }
        let d = self.gate_distance(m);
        r += cfg.k_progress * (self.d_prev - d);
        self.d_prev = d;
        if cfg.k_near > 0.0 && (self.end == End::None || self.end == End::Finish) {
            let c = m.clearance(&self.st.p);
            if c < cfg.d_near {
                r -= cfg.k_near * (1.0 - c.max(0.0) / cfg.d_near).powi(2);
            }
        }
        let w = [self.st.w[0].to_radians(), self.st.w[1].to_radians(), self.st.w[2].to_radians()];
        r -= cfg.k_rate * dot(&w, &w);
        let da: f64 = (0..4).map(|k| (a[k] - self.a_prev[k]).powi(2)).sum();
        r -= cfg.k_smooth * da;
        r -= cfg.k_time;
        self.a_prev = a;
        self.steps += 1;
        let term = self.end != End::None;
        if term && self.end != End::Finish {
            r -= cfg.k_crash;
        }
        let trunc = !term && self.steps >= cfg.max_steps;
        if trunc {
            self.end = End::Time;
        }
        self.ret += r;
        (r, term, trunc)
    }

    pub fn close_episode(&mut self) {
        self.finished.push(Episode {
            ret: self.ret,
            steps: self.steps,
            gates: self.gates,
            laps: self.laps,
            best_lap: self.best_lap,
            race_time: self.race_time,
            bat_end: self.st.bat,
            end: self.end,
        });
    }

    pub fn observe(&self, m: &World, o: &mut [f32]) {
        let q = &self.st.q;
        let rt = rotate(q, &[1.0, 0.0, 0.0]);
        let up = rotate(q, &[0.0, 1.0, 0.0]);
        let fw = rotate(q, &[0.0, 0.0, 1.0]);
        let p = &self.st.p;
        let body = |v: &V3| [dot(v, &rt), dot(v, &up), dot(v, &fw)];
        let mut k = 0;
        let mut put = |x: f64, k: &mut usize| {
            o[*k] = x as f32;
            *k += 1;
        };
        for x in body(&self.st.v) {
            put(x / 10.0, &mut k);
        }
        for e in [&rt, &up, &fw] {
            for x in e.iter() {
                put(*x, &mut k);
            }
        }
        for x in self.st.w {
            put(x / 200.0, &mut k);
        }
        for x in self.a_prev {
            put(x, &mut k);
        }
        let n = m.gates.len();
        for j in 0..GATES_IN_OBS {
            let g = &m.gates[(self.gate + j) % n];
            for e in &g.corners {
                for x in body(&[e[0] - p[0], e[1] - p[1], e[2] - p[2]]) {
                    put(x / 10.0, &mut k);
                }
            }
        }
        put((p[1] - m.terrain.height(p[0], p[2])) / 10.0, &mut k);
        put(self.st.bat, &mut k);
        put(self.laps_target.saturating_sub(self.laps) as f64 / RACE_LAPS as f64, &mut k);
        put(self.gate as f64 / n as f64, &mut k);
        debug_assert_eq!(k, OBS);
    }
}

pub struct Batch {
    pub world: Arc<World>,
    pub cfg: Cfg,
    pub envs: Vec<Env>,
}
