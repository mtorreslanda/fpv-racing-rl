//! A race track: terrain height map, solid obstacles, gates in race order, flight zone and a reference line.
//! Tracks are JSON files (see `python/make_track.py`); the height map is a raw little-endian f32 file next to it.

use crate::physics::{dot, norm, rotate, Q, V3};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct JTerrain {
    file: String,
    origin: V3,
    size: V3,
    res: usize,
}
#[derive(Deserialize)]
struct JZone {
    min: V3,
    max: V3,
}
#[derive(Deserialize)]
struct JGate {
    id: i64,
    center: V3,
    right: V3,
    up: V3,
    forward: V3,
    half_width: f64,
    half_height: f64,
}
#[derive(Deserialize)]
struct JBox {
    c: V3,
    s: V3,
    q: Q,
}
#[derive(Deserialize)]
struct JCapsule {
    a: V3,
    b: V3,
    r: f64,
}
#[derive(Deserialize)]
struct JSphere {
    c: V3,
    r: f64,
}
#[derive(Deserialize)]
struct JTrack {
    terrain: JTerrain,
    zone: JZone,
    start: V3,
    drone_radius: f64,
    gates: Vec<JGate>,
    boxes: Vec<JBox>,
    capsules: Vec<JCapsule>,
    spheres: Vec<JSphere>,
    line: Vec<V3>,
}

pub struct Terrain {
    o: V3,
    dx: f64,
    dz: f64,
    res: usize,
    h: Vec<f32>,
}

impl Terrain {
    /// Bilinear (row = z, column = x; clamped to res − 1.001).
    pub fn height(&self, x: f64, z: f64) -> f64 {
        let lim = self.res as f64 - 1.001;
        let fx = ((x - self.o[0]) / self.dx).clamp(0.0, lim);
        let fz = ((z - self.o[2]) / self.dz).clamp(0.0, lim);
        let (i, j) = (fz.floor() as usize, fx.floor() as usize);
        let (a, b) = (fz - i as f64, fx - j as f64);
        let n = self.res;
        let h = |r: usize, c: usize| self.h[r * n + c] as f64;
        h(i, j) * (1.0 - a) * (1.0 - b) + h(i, j + 1) * (1.0 - a) * b + h(i + 1, j) * a * (1.0 - b) + h(i + 1, j + 1) * a * b
    }
}

pub enum Shape {
    /// oriented box: centre, half extents, inverse rotation (world -> local)
    Box { c: V3, s: V3, qi: Q },
    Capsule { a: V3, b: V3, r: f64 },
    Sphere { c: V3, r: f64 },
}

impl Shape {
    /// Signed distance from a point to the shape (negative inside).
    pub fn distance(&self, p: &V3) -> f64 {
        match self {
            Shape::Box { c, s, qi } => {
                let l = rotate(qi, &[p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
                let d = [l[0].abs() - s[0], l[1].abs() - s[1], l[2].abs() - s[2]];
                let out = [d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)];
                norm(&out) + d[0].max(d[1]).max(d[2]).min(0.0)
            }
            Shape::Capsule { a, b, r } => {
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let t = (dot(&ap, &ab) / dot(&ab, &ab).max(1e-12)).clamp(0.0, 1.0);
                let q = [a[0] + t * ab[0] - p[0], a[1] + t * ab[1] - p[1], a[2] + t * ab[2] - p[2]];
                norm(&q) - r
            }
            Shape::Sphere { c, r } => norm(&[p[0] - c[0], p[1] - c[1], p[2] - c[2]]) - r,
        }
    }

    /// Bounding box in xz (for the grid).
    fn xz(&self) -> (f64, f64, f64, f64) {
        match self {
            Shape::Box { c, s, .. } => {
                let r = norm(s);
                (c[0] - r, c[0] + r, c[2] - r, c[2] + r)
            }
            Shape::Capsule { a, b, r } => (a[0].min(b[0]) - r, a[0].max(b[0]) + r, a[2].min(b[2]) - r, a[2].max(b[2]) + r),
            Shape::Sphere { c, r } => (c[0] - r, c[0] + r, c[2] - r, c[2] + r),
        }
    }
}

/// Each grid cell keeps every shape within drone radius + GRID_MARGIN of it: distances are exact up to that
/// clearance (collisions and the proximity penalty).
pub const GRID_MARGIN: f64 = 0.6;

pub struct Gate {
    pub id: i64,
    pub c: V3,
    pub right: V3,
    pub up: V3,
    pub fwd: V3,
    pub hw: f64,
    pub hh: f64,
    pub corners: [V3; 4],
}

pub struct World {
    pub terrain: Terrain,
    pub zmin: V3,
    pub zmax: V3,
    pub start: V3,
    pub radius: f64,
    pub gates: Vec<Gate>,
    pub shapes: Vec<Shape>,
    cell: f64,
    gx0: f64,
    gz0: f64,
    gnx: usize,
    gnz: usize,
    grid: Vec<Vec<u32>>,
    pub line: Vec<V3>,
    pub line_tan: Vec<V3>,
    /// for each point of the line, the next gate ahead
    pub line_gate: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum End {
    None = 0,
    Obstacle = 1,
    Terrain = 2,
    Zone = 3,
    Time = 4,
    /// race finished (race mode)
    Finish = 5,
    /// inside the safety margin (counts as a crash in training)
    Margin = 6,
}

impl World {
    pub fn load(path: &str) -> Result<World, String> {
        let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let j: JTrack = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        let bin = Path::new(path).parent().unwrap_or(Path::new(".")).join(&j.terrain.file);
        let bytes = std::fs::read(&bin).map_err(|e| format!("{}: {e}", bin.display()))?;
        let h: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let res = j.terrain.res;
        if h.len() != res * res {
            return Err(format!("terrain: {} values, expected {}", h.len(), res * res));
        }
        let terrain = Terrain {
            o: j.terrain.origin,
            dx: j.terrain.size[0] / (res as f64 - 1.0),
            dz: j.terrain.size[2] / (res as f64 - 1.0),
            res,
            h,
        };
        let mut shapes = Vec::new();
        for b in &j.boxes {
            shapes.push(Shape::Box { c: b.c, s: b.s, qi: [-b.q[0], -b.q[1], -b.q[2], b.q[3]] });
        }
        for k in &j.capsules {
            shapes.push(Shape::Capsule { a: k.a, b: k.b, r: k.r });
        }
        for e in &j.spheres {
            shapes.push(Shape::Sphere { c: e.c, r: e.r });
        }
        // xz grid over the zone, with the drone radius as margin
        let cell = 4.0;
        let (gx0, gz0) = (j.zone.min[0] - cell, j.zone.min[2] - cell);
        let gnx = ((j.zone.max[0] - gx0) / cell) as usize + 2;
        let gnz = ((j.zone.max[2] - gz0) / cell) as usize + 2;
        let mut grid = vec![Vec::new(); gnx * gnz];
        let m = j.drone_radius + GRID_MARGIN;
        for (k, f) in shapes.iter().enumerate() {
            let (x0, x1, z0, z1) = f.xz();
            let ci = |x: f64, o: f64, n: usize| (((x - o) / cell).floor().max(0.0) as usize).min(n - 1);
            for iz in ci(z0 - m, gz0, gnz)..=ci(z1 + m, gz0, gnz) {
                for ix in ci(x0 - m, gx0, gnx)..=ci(x1 + m, gx0, gnx) {
                    grid[iz * gnx + ix].push(k as u32);
                }
            }
        }
        let gates: Vec<Gate> = j
            .gates
            .iter()
            .map(|p| {
                let e = |sx: f64, sy: f64| {
                    [
                        p.center[0] + sx * p.half_width * p.right[0] + sy * p.half_height * p.up[0],
                        p.center[1] + sx * p.half_width * p.right[1] + sy * p.half_height * p.up[1],
                        p.center[2] + sx * p.half_width * p.right[2] + sy * p.half_height * p.up[2],
                    ]
                };
                Gate {
                    id: p.id,
                    c: p.center,
                    right: p.right,
                    up: p.up,
                    fwd: p.forward,
                    hw: p.half_width,
                    hh: p.half_height,
                    corners: [e(-1.0, -1.0), e(1.0, -1.0), e(1.0, 1.0), e(-1.0, 1.0)],
                }
            })
            .collect();
        // reference line: tangents and next gate
        let r = &j.line;
        let n = r.len();
        let line_tan: Vec<V3> = (0..n)
            .map(|i| {
                let (a, b) = (r[i.saturating_sub(1)], r[(i + 1).min(n - 1)]);
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let l = norm(&d).max(1e-9);
                [d[0] / l, d[1] / l, d[2] / l]
            })
            .collect();
        let gate_idx: Vec<usize> = gates
            .iter()
            .map(|p| {
                (0..n)
                    .min_by(|&a, &b| {
                        let da = norm(&[r[a][0] - p.c[0], r[a][1] - p.c[1], r[a][2] - p.c[2]]);
                        let db = norm(&[r[b][0] - p.c[0], r[b][1] - p.c[1], r[b][2] - p.c[2]]);
                        da.partial_cmp(&db).unwrap()
                    })
                    .unwrap()
            })
            .collect();
        let line_gate: Vec<usize> = (0..n).map(|i| (0..gates.len()).find(|&g| gate_idx[g] > i).unwrap_or(0)).collect();
        Ok(World {
            terrain,
            zmin: j.zone.min,
            zmax: j.zone.max,
            start: j.start,
            radius: j.drone_radius,
            gates,
            shapes,
            cell,
            gx0,
            gz0,
            gnx,
            gnz,
            grid,
            line: j.line,
            line_tan,
            line_gate,
        })
    }

    /// Does the drone (a sphere) collide at p? Returns why.
    pub fn collision(&self, p: &V3) -> End {
        for k in 0..3 {
            if p[k] < self.zmin[k] || p[k] > self.zmax[k] {
                return End::Zone;
            }
        }
        if p[1] - self.radius < self.terrain.height(p[0], p[2]) {
            return End::Terrain;
        }
        let ix = ((p[0] - self.gx0) / self.cell) as usize;
        let iz = ((p[2] - self.gz0) / self.cell) as usize;
        if ix < self.gnx && iz < self.gnz {
            for &k in &self.grid[iz * self.gnx + ix] {
                if self.shapes[k as usize].distance(p) < self.radius {
                    return End::Obstacle;
                }
            }
        }
        End::None
    }

    /// Clearance: what is left between the drone's surface and the terrain or the nearest obstacle
    /// (saturated at GRID_MARGIN).
    pub fn clearance(&self, p: &V3) -> f64 {
        let alt = p[1] - self.terrain.height(p[0], p[2]);
        let d = self.obstacle_distance(p).min(alt);
        (d - self.radius).min(GRID_MARGIN)
    }

    /// Smallest signed distance to the obstacles of the cell (for tests).
    pub fn obstacle_distance(&self, p: &V3) -> f64 {
        let ix = ((p[0] - self.gx0) / self.cell) as usize;
        let iz = ((p[2] - self.gz0) / self.cell) as usize;
        let mut d = f64::INFINITY;
        if ix < self.gnx && iz < self.gnz {
            for &k in &self.grid[iz * self.gnx + ix] {
                d = d.min(self.shapes[k as usize].distance(p));
            }
        }
        d
    }

    /// Does segment a -> b cross gate g in its direction and inside its opening (shrunk by `margin` per side)?
    pub fn crosses(&self, g: usize, a: &V3, b: &V3, margin: f64) -> bool {
        let gt = &self.gates[g];
        let sa = dot(&[a[0] - gt.c[0], a[1] - gt.c[1], a[2] - gt.c[2]], &gt.fwd);
        let sb = dot(&[b[0] - gt.c[0], b[1] - gt.c[1], b[2] - gt.c[2]], &gt.fwd);
        if !(sa < 0.0 && sb >= 0.0) {
            return false;
        }
        let t = sa / (sa - sb);
        let x = [
            a[0] + t * (b[0] - a[0]) - gt.c[0],
            a[1] + t * (b[1] - a[1]) - gt.c[1],
            a[2] + t * (b[2] - a[2]) - gt.c[2],
        ];
        dot(&x, &gt.right).abs() <= gt.hw - margin && dot(&x, &gt.up).abs() <= gt.hh - margin
    }
}
