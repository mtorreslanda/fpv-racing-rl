"""PPO on fpv_sim. One file, CleanRL style.

- Environment: fpv_sim.Racing (Rust), thousands of races in parallel, policy at 50 Hz, physics randomised per episode.
- Policy: MLP 59 -> 128 -> 128 -> 4 (tanh), Gaussian with a learned log sigma; the throttle mean starts at hover.
- Value: MLP 59 -> 256 -> 256 -> 1.
- Observations normalised (running mean and variance); rewards scaled by the sigma of the discounted return.
- Truncation is bootstrapped with V(final_obs), never treated as a crash.
- Evaluation every --eval_every iterations: nominal physics, delay 2, from the start pad, mean action, full race.
  The best checkpoint = most races finished, then the fastest median race time.

Curriculum used for the published policy (see README):
  1. laps:  python train/ppo.py --name laps --minutes 15
  2. races: python train/ppo.py --name races --minutes 30 --init runs/laps/last.pt --warmup 10 \
              --cfg '{"race": true, "p_start": 0.3, "bat_min": 0.3, "margin": 0.1, "max_steps": 6000}' --gamma 0.999
Stop cleanly at any time by creating the file runs/<name>/STOP.
"""
import argparse, csv, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import fpv_sim

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRACK = os.path.join(ROOT, "tracks", "hillside.json")
HOVER = torch.tensor([-0.399, 0.0, 0.0, 0.0])


class RunningNorm:
    def __init__(self, d, dev):
        self.m = torch.zeros(d, device=dev); self.v = torch.ones(d, device=dev); self.n = 1e-4; self.frozen = False

    def update(self, x):
        if self.frozen:
            return
        bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.m; tot = self.n + bn
        self.m = self.m + d * bn / tot
        self.v = (self.v * self.n + bv * bn + d * d * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return ((x - self.m) / torch.sqrt(self.v + 1e-8)).clamp(-10, 10)


def mlp(i, h, o, out_std=1.0):
    layers = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for k, c in enumerate(c for c in layers if isinstance(c, nn.Linear)):
        nn.init.orthogonal_(c.weight, math.sqrt(2) if k < 2 else out_std); nn.init.zeros_(c.bias)
    return layers


class Agent(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pi = mlp(d, 128, 4, 0.01)
        self.v = mlp(d, 256, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((4,), -1.0))
        self.register_buffer("hover", HOVER.clone())

    def mean(self, o):
        return self.pi(o) + self.hover

    def dist(self, o):
        return torch.distributions.Normal(self.mean(o), self.log_std.exp().expand(o.shape[0], 4))


def evaluate(ag, norm, dev, track=TRACK, n=32, steps=8000, margin=0.3):
    """A 3-lap race from the start pad, full battery, nominal physics, mean action."""
    E = fpv_sim.Racing(track, n, 12345, {"randomize": False, "delay": 2, "p_start": 1.0, "max_steps": steps,
                                         "gate_margin": margin, "bat_fixed": 1.0, "race": True})
    o = torch.as_tensor(E.reset(), device=dev); alive = np.ones(n, bool)
    with torch.no_grad():
        for _ in range(steps):
            a = ag.mean(norm(o)).clamp(-1, 1).cpu().numpy().astype(np.float32)
            o, r, te, tr, _ = E.step(a); o = torch.as_tensor(o, device=dev)
            alive &= ~(te | tr)
            if not alive.any():
                break
    ep = E.episodes(); bl, laps, end, rt = ep["best_lap"], ep["laps"], ep["end"], ep["race_time"]
    fin = lambda x, f: float(f(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")
    return {"episodes": int(len(end)), "crash_frac": float(np.mean((end < 4) | (end == 6))),
            "laps_mean": float(laps.mean()), "best_lap": fin(bl, np.min), "median_lap": fin(bl, np.median),
            "gates_mean": float(ep["gates"].mean()), "race_frac": float(np.mean(np.isfinite(rt))),
            "race_median": fin(rt, np.median), "bat_end_median": float(np.median(ep["bat_end"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--track", default=TRACK)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cfg", type=str, default="{}", help="JSON with environment keys (race, margin, k_near...)")
    ap.add_argument("--init", type=str, default="", help="checkpoint .pt to start from (its normaliser is then frozen)")
    ap.add_argument("--warmup", type=int, default=0, help="first iterations train only the critic (policy frozen)")
    a = ap.parse_args()

    out = os.path.join(ROOT, "runs", a.name); os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_env = json.loads(a.cfg)
    E = fpv_sim.Racing(a.track, a.envs, a.seed, cfg_env); D = E.obs_dim
    json.dump({**vars(a), "obs_dim": D, "device": str(dev), "started": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(out, "config.json"), "w"), indent=1)

    ag = Agent(D).to(dev); norm = RunningNorm(D, dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        ag.load_state_dict(ck["agent"]); norm.m = ck["norm_m"].to(dev); norm.v = ck["norm_v"].to(dev)
        # a moving normaliser under a policy that was trained with the old one makes it crash before it can adapt
        norm.n = 1e15; norm.frozen = True
        print(f"starting from {a.init}", flush=True)
    opt = torch.optim.Adam(ag.parameters(), lr=a.lr, eps=1e-5)
    ret_run = np.zeros(a.envs); rnorm = RunningNorm(1, dev)

    N, T = a.envs, a.steps
    O = torch.zeros(T, N, D, device=dev); A = torch.zeros(T, N, 4, device=dev); LP = torch.zeros(T, N, device=dev)
    R = torch.zeros(T, N, device=dev); DN = torch.zeros(T, N, device=dev); V = torch.zeros(T, N, device=dev)

    o = torch.as_tensor(E.reset(), device=dev); norm.update(o)
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    wr = csv.writer(log); wr.writerow(["iter", "steps", "sec", "sps", "ep_return", "ep_len", "ep_gates", "crash_frac", "ep_laps",
                                       "std", "v_loss", "kl", "ev_best_lap", "ev_median_lap", "ev_crash", "ev_laps",
                                       "ev_race_frac", "ev_race_median", "ev_bat_end"])
    best = (-1.0,); t0 = time.time(); it = 0; total = 0
    stop = os.path.join(out, "STOP")
    while time.time() - t0 < a.minutes * 60 and not os.path.exists(stop):
        it += 1
        with torch.no_grad():
            for t in range(T):
                on = norm(o); d = ag.dist(on); act = d.sample()
                O[t] = on; A[t] = act; LP[t] = d.log_prob(act).sum(-1); V[t] = ag.v(on).squeeze(-1)
                o2, r, te, tr, of = E.step(act.clamp(-1, 1).cpu().numpy().astype(np.float32))
                r = r.astype(np.float64); done = te | tr
                ret_run = ret_run * a.gamma + r; rnorm.update(torch.as_tensor(ret_run, device=dev, dtype=torch.float32)[:, None])
                ret_run[done] = 0.0
                rt = torch.as_tensor(r, device=dev, dtype=torch.float32) / torch.sqrt(rnorm.v[0] + 1e-8)
                if tr.any():   # bootstrap truncation with the value of the final state
                    idx = torch.as_tensor(np.where(tr)[0], device=dev)
                    rt[idx] += a.gamma * ag.v(norm(torch.as_tensor(of[tr], device=dev))).squeeze(-1)
                R[t] = rt; DN[t] = torch.as_tensor(done, device=dev, dtype=torch.float32)
                o = torch.as_tensor(o2, device=dev); norm.update(o)
            adv = torch.zeros_like(R); last = torch.zeros(N, device=dev); v_next = ag.v(norm(o)).squeeze(-1)
            for t in reversed(range(T)):   # GAE
                nd = 1.0 - DN[t]
                delta = R[t] + a.gamma * v_next * nd - V[t]
                last = delta + a.gamma * a.lam * nd * last
                adv[t] = last; v_next = V[t]
            ret = adv + V
        b_o, b_a, b_lp, b_adv, b_ret = O.reshape(-1, D), A.reshape(-1, 4), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        B = b_o.shape[0]; mb = B // a.minibatches; kls = []
        for _ in range(a.epochs):
            perm = torch.randperm(B, device=dev)
            for k in range(a.minibatches):
                j = perm[k * mb:(k + 1) * mb]
                d = ag.dist(b_o[j]); lp = d.log_prob(b_a[j]).sum(-1); ratio = (lp - b_lp[j]).exp()
                ad = (b_adv[j] - b_adv[j].mean()) / (b_adv[j].std() + 1e-8)
                pl = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                vl = 0.5 * ((ag.v(b_o[j]).squeeze(-1) - b_ret[j]) ** 2).mean()
                loss = (0.5 * vl) if it <= a.warmup else (pl + 0.5 * vl)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ag.parameters(), 1.0); opt.step()
                with torch.no_grad(): kls.append(((ratio - 1) - (lp - b_lp[j])).mean().item())
        total += N * T
        ep = E.episodes(); end = ep["end"]; has = len(end) > 0
        row = [it, total, round(time.time() - t0), round(total / (time.time() - t0)),
               round(float(ep["return"].mean()), 2) if has else "", round(float(ep["steps"].mean()), 1) if has else "",
               round(float(ep["gates"].mean()), 2) if has else "", round(float(np.mean((end < 4) | (end == 6))), 3) if has else "",
               round(float(ep["laps"].mean()), 3) if has else "", round(float(ag.log_std.detach().exp().mean()), 3),
               round(vl.item(), 4), round(float(np.mean(kls)), 4)]
        if it % a.eval_every == 0:
            ev = evaluate(ag, norm, dev, a.track)
            row += [ev["best_lap"], ev["median_lap"], ev["crash_frac"], ev["laps_mean"], ev["race_frac"], ev["race_median"], ev["bat_end_median"]]
            fin = lambda x: -x if math.isfinite(x) else -float("inf")
            # finished races first, then faster races; before any race is finished: more laps, then a faster lap
            key = (ev["race_frac"], fin(ev["race_median"]), ev["laps_mean"], fin(ev["best_lap"]))
            state = {"agent": ag.state_dict(), "norm_m": norm.m, "norm_v": norm.v, "iter": it, "steps": total, "eval": ev}
            torch.save(state, os.path.join(out, "last.pt")); torch.save(state, os.path.join(out, f"eval_{it:05d}.pt"))
            if key > best:
                best = key; torch.save(state, os.path.join(out, "best.pt"))
            print(f"[{it}] {total/1e6:.0f} M steps · {row[3]} steps/s · train: gates {row[6]} crash {row[7]} · EVAL: laps "
                  f"{ev['laps_mean']:.2f}, crash {ev['crash_frac']:.2f}, best lap {ev['best_lap']:.2f} s · race finished "
                  f"{ev['race_frac']:.2f}, median {ev['race_median']:.2f} s, battery left {ev['bat_end_median']*100:.0f} %", flush=True)
        else:
            row += [""] * 7
        wr.writerow(row); log.flush()
    print("done", time.strftime("%H:%M:%S"), "· best:", best, flush=True)


if __name__ == "__main__":
    main()
