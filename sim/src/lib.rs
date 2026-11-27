//! fpv_sim: a fast quadcopter racing simulator for reinforcement learning, with Python bindings.
//! physics: identified grey-box model · world: track, collisions, gates · env: a batch of N races, in parallel.

pub mod env;
pub mod physics;
pub mod residual;
pub mod world;

use env::{Batch, Cfg, Env, OBS};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use physics::{step, Params, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::Arc;
use world::World;

/// Open loop, like `fpv_sim_ref.step` in a loop: state0 = [p(3), v(3), q(4), w(3), bat];
/// sticks = (N, 4) (thr, yaw, pitch, roll); prev0 = the sticks of the step before the first.
/// Returns (N, 14): the state after each step. Model v0.2, no ground.
#[pyfunction]
fn simulate_open_loop<'py>(
    py: Python<'py>,
    state0: PyReadonlyArray1<'py, f64>,
    sticks: PyReadonlyArray2<'py, f64>,
    prev0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut st = to_state(state0.as_slice()?);
    let a = sticks.as_array();
    let pp = prev0.as_slice()?;
    let mut prev = [pp[0], pp[1], pp[2], pp[3]];
    let par = Params::v02();
    let n = a.nrows();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((n, 14));
    for k in 0..n {
        let inp = [a[[k, 0]], a[[k, 1]], a[[k, 2]], a[[k, 3]]];
        step(&mut st, &inp, &prev, &par);
        prev = inp;
        for (j, x) in from_state(&st).iter().enumerate() {
            out[[k, j]] = *x;
        }
    }
    Ok(out.into_pyarray(py))
}

fn to_state(e: &[f64]) -> State {
    State { p: [e[0], e[1], e[2]], v: [e[3], e[4], e[5]], q: [e[6], e[7], e[8], e[9]], w: [e[10], e[11], e[12]], bat: e[13] }
}
fn from_state(s: &State) -> [f64; 14] {
    [s.p[0], s.p[1], s.p[2], s.v[0], s.v[1], s.v[2], s.q[0], s.q[1], s.q[2], s.q[3], s.w[0], s.w[1], s.w[2], s.bat]
}

fn read_cfg(d: Option<&Bound<'_, PyDict>>) -> PyResult<Cfg> {
    let mut c = Cfg::default();
    if let Some(d) = d {
        for (k, v) in d.iter() {
            let k: String = k.extract()?;
            match k.as_str() {
                "decim" => c.decim = v.extract()?,
                "max_steps" => c.max_steps = v.extract()?,
                "randomize" => c.randomize = v.extract()?,
                "delay" => c.delay = v.extract()?,
                "k_progress" => c.k_progress = v.extract()?,
                "k_gate" => c.k_gate = v.extract()?,
                "k_rate" => c.k_rate = v.extract()?,
                "k_smooth" => c.k_smooth = v.extract()?,
                "k_crash" => c.k_crash = v.extract()?,
                "v_init_max" => c.v_init_max = v.extract()?,
                "p_start" => c.p_start = v.extract()?,
                "gate_margin" => c.gate_margin = v.extract()?,
                "k_near" => c.k_near = v.extract()?,
                "d_near" => c.d_near = v.extract()?,
                "bat_min" => c.bat_min = v.extract()?,
                "bat_fixed" => c.bat_fixed = v.extract()?,
                "race" => c.race = v.extract()?,
                "k_finish" => c.k_finish = v.extract()?,
                "margin" => c.margin = v.extract()?,
                "k_time" => c.k_time = v.extract()?,
                "res_min" => c.res_min = v.extract()?,
                "res_max" => c.res_max = v.extract()?,
                "bias_n" => c.bias_n = v.extract()?,
                "model" => {}
                _ => return Err(PyValueError::new_err(format!("unknown config key: {k}"))),
            }
        }
    }
    if c.delay < 1 || c.delay > 5 {
        return Err(PyValueError::new_err("delay outside 1..=5"));
    }
    Ok(c)
}

/// A batch of N independent races on one track. Actions (N, 4) = sticks (thr, yaw, pitch, roll) in [−1, 1].
/// step() resets the finished environments and already returns the new episode's observation.
#[pyclass]
struct Racing {
    b: Batch,
}

#[pymethods]
impl Racing {
    #[new]
    #[pyo3(signature = (track, n, seed=0, cfg=None))]
    fn new(track: &str, n: usize, seed: u64, cfg: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let world = Arc::new(World::load(track).map_err(PyValueError::new_err)?);
        let mut model = None;
        if let Some(d) = cfg {
            if let Some(v) = d.get_item("model")? {
                let path: String = v.extract()?;
                if !path.is_empty() {
                    model = Some(Arc::new(residual::Model::load(&path).map_err(PyValueError::new_err)?));
                }
            }
        }
        let cfg = read_cfg(cfg)?;
        let envs = (0..n)
            .map(|i| Env::new(seed.wrapping_mul(0x2545F4914F6CDD1D) ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        Ok(Racing { b: Batch { world, model, cfg, envs } })
    }

    #[getter]
    fn n(&self) -> usize {
        self.b.envs.len()
    }
    #[getter]
    fn obs_dim(&self) -> usize {
        OBS
    }
    #[getter]
    fn n_gates(&self) -> usize {
        self.b.world.gates.len()
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let (m, c) = (&self.b.world, &self.b.cfg);
        self.b.envs.par_iter_mut().for_each(|e| e.reset(m, c));
        self.obs(py)
    }

    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.b.envs.len();
        let mut o = vec![0f32; n * OBS];
        let m = &self.b.world;
        o.par_chunks_mut(OBS).zip(self.b.envs.par_iter()).for_each(|(oi, e)| e.observe(m, oi));
        numpy::ndarray::Array2::from_shape_vec((n, OBS), o).unwrap().into_pyarray(py)
    }

    /// Returns (obs, reward, terminated, truncated, final_obs). final_obs = the observation at the end (before the
    /// reset) for environments that terminated or were truncated, zeros elsewhere: for bootstrapping truncation.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray2<f32>>)>
    {
        let n = self.b.envs.len();
        let a = actions.as_array();
        if a.shape() != [n, 4] {
            return Err(PyValueError::new_err(format!("actions: shape {:?}, expected ({n}, 4)", a.shape())));
        }
        let a: Vec<[f64; 4]> = (0..n).map(|i| [a[[i, 0]] as f64, a[[i, 1]] as f64, a[[i, 2]] as f64, a[[i, 3]] as f64]).collect();
        let mut r = vec![0f32; n];
        let mut te = vec![false; n];
        let mut tr = vec![false; n];
        let mut o = vec![0f32; n * OBS];
        let mut of = vec![0f32; n * OBS];
        {
            let (m, c) = (&self.b.world, &self.b.cfg);
            let md = self.b.model.as_deref();
            let envs = &mut self.b.envs;
            py.detach(|| {
                envs.par_iter_mut()
                    .zip(a.par_iter())
                    .zip(r.par_iter_mut())
                    .zip(te.par_iter_mut())
                    .zip(tr.par_iter_mut())
                    .zip(o.par_chunks_mut(OBS))
                    .zip(of.par_chunks_mut(OBS))
                    .for_each(|((((((e, ai), ri), tei), tri), oi), ofi)| {
                        let (rr, t1, t2) = e.step(m, md, c, *ai);
                        *ri = rr as f32;
                        *tei = t1;
                        *tri = t2;
                        if t1 || t2 {
                            e.observe(m, ofi);
                            e.close_episode();
                            e.reset(m, c);
                        }
                        e.observe(m, oi);
                    });
            });
        }
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, OBS), o).unwrap().into_pyarray(py),
            r.into_pyarray(py),
            te.into_pyarray(py),
            tr.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, OBS), of).unwrap().into_pyarray(py),
        ))
    }

    /// Episodes finished since the last call: dict of arrays (return, steps, gates, laps, best_lap [NaN if none],
    /// end [1 obstacle, 2 terrain, 3 zone, 4 time, 5 finish, 6 safety margin], race_time, bat_end).
    fn episodes<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut v = Vec::new();
        for e in self.b.envs.iter_mut() {
            v.append(&mut e.finished);
        }
        let d = PyDict::new(py);
        d.set_item("return", v.iter().map(|x| x.ret).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("steps", v.iter().map(|x| x.steps as i64).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("gates", v.iter().map(|x| x.gates as i64).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("laps", v.iter().map(|x| x.laps as i64).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("best_lap", v.iter().map(|x| x.best_lap).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("end", v.iter().map(|x| x.end as i64).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("race_time", v.iter().map(|x| x.race_time).collect::<Vec<_>>().into_pyarray(py))?;
        d.set_item("bat_end", v.iter().map(|x| x.bat_end).collect::<Vec<_>>().into_pyarray(py))?;
        Ok(d)
    }

    // ---- tests and external control loops ----

    /// Sets environment i: state [14], command that fills the history, next gate, delay, and whether the
    /// physics is the training one (v0.4: battery + high-speed drag) or v0.2 (the equivalence tests').
    #[pyo3(signature = (i, state, cmd, gate, delay=2, battery=false))]
    fn set_state(&mut self, i: usize, state: PyReadonlyArray1<'_, f64>, cmd: PyReadonlyArray1<'_, f64>, gate: usize, delay: usize, battery: bool) -> PyResult<()> {
        let c = cmd.as_slice()?;
        let m = self.b.world.clone();
        let e = self.b.envs.get_mut(i).ok_or_else(|| PyValueError::new_err("i out of range"))?;
        e.par = if battery { Params::v04() } else { Params::v02() };
        e.set(&m, to_state(state.as_slice()?), [c[0], c[1], c[2], c[3]], gate, delay);
        Ok(())
    }

    /// One decision of environment i without automatic reset: (reward, terminated, truncated).
    fn step_one(&mut self, i: usize, a: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, bool, bool)> {
        let a = a.as_slice()?;
        let m = self.b.world.clone();
        let c = self.b.cfg.clone();
        let md = self.b.model.clone();
        Ok(self.b.envs[i].step(&m, md.as_deref(), &c, [a[0], a[1], a[2], a[3]]))
    }

    fn state<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyArray1<f64>> {
        from_state(&self.b.envs[i].st).to_vec().into_pyarray(py)
    }

    /// Dict with gate, gates, laps, last_lap, best_lap, time, end, race_time, battery, laps_target, delay of env i.
    fn info<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let e = &self.b.envs[i];
        let d = PyDict::new(py);
        d.set_item("gate", e.gate)?;
        d.set_item("gates", e.gates)?;
        d.set_item("laps", e.laps)?;
        d.set_item("last_lap", e.last_lap)?;
        d.set_item("best_lap", e.best_lap)?;
        d.set_item("time", e.time)?;
        d.set_item("end", e.end as i64)?;
        d.set_item("race_time", e.race_time)?;
        d.set_item("battery", e.st.bat)?;
        d.set_item("laps_target", e.laps_target)?;
        d.set_item("delay", e.delay)?;
        Ok(d)
    }

    /// Signed distance to obstacles, terrain height and collision code at points (M, 3) (world tests).
    fn probe<'py>(&self, py: Python<'py>, p: PyReadonlyArray2<'py, f64>) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>) {
        let p = p.as_array();
        let m = &self.b.world;
        let (mut d, mut h, mut f) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..p.nrows() {
            let q = [p[[k, 0]], p[[k, 1]], p[[k, 2]]];
            d.push(m.obstacle_distance(&q));
            h.push(m.terrain.height(q[0], q[2]));
            f.push(m.collision(&q) as i64);
        }
        (d.into_pyarray(py), h.into_pyarray(py), f.into_pyarray(py))
    }
}

/// Open loop with the base + residual model (equivalence with python/residual.py): state0 [14], sticks (N, 4),
/// prev0 [4], heights (N) over the ground, the model JSON. No battery drain (as the reference).
#[pyfunction]
fn simulate_open_loop_model<'py>(
    py: Python<'py>,
    state0: PyReadonlyArray1<'py, f64>,
    sticks: PyReadonlyArray2<'py, f64>,
    prev0: PyReadonlyArray1<'py, f64>,
    heights: PyReadonlyArray1<'py, f64>,
    model: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let md = residual::Model::load(model).map_err(PyValueError::new_err)?;
    let mut st = to_state(state0.as_slice()?);
    let a = sticks.as_array();
    let hs = heights.as_slice()?;
    let pp = prev0.as_slice()?;
    let mut prev = [pp[0], pp[1], pp[2], pp[3]];
    let par = Params::v02();
    let n = a.nrows();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((n, 14));
    for k in 0..n {
        let inp = [a[[k, 0]], a[[k, 1]], a[[k, 2]], a[[k, 3]]];
        residual::step_residual(&mut st, &inp, &prev, &par, &md, hs[k], 1.0, 1.0, 1.0, &[0.0; 3]);
        prev = inp;
        for (j, x) in from_state(&st).iter().enumerate() {
            out[[k, j]] = *x;
        }
    }
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn fpv_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_open_loop, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_open_loop_model, m)?)?;
    m.add_class::<Racing>()?;
    Ok(())
}
