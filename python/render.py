"""Figures and animations for the README, from the synthetic track and the training runs.

  python python/render.py track                       -> media/track.png        the track from above
  python python/render.py learning runs/laps runs/races -> media/learning.png   one race per saved checkpoint, crashes as X
  python python/render.py race runs/races/best.pt      -> media/race.gif + media/race-telemetry.png
  python python/render.py curve runs/laps runs/races    -> media/curve.png       best lap and finished races over training
"""
import csv, glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "train"))
TRACK = os.path.join(ROOT, "tracks", "hillside.json")
MEDIA = os.path.join(ROOT, "media")
T = json.load(open(TRACK))
H = np.fromfile(os.path.join(ROOT, "tracks", T["terrain"]["file"]), "<f4").reshape(T["terrain"]["res"], -1)
BG, INK, GREY, AMBER, RED = "#0B0E0F", "#EAF2EE", "#7F8C87", "#FFB21F", "#FF4B3A"
SPEED = LinearSegmentedColormap.from_list("v", ["#2F7F95", "#3FA39B", "#7CBF7A", "#C9C45A", "#FFB21F", "#FF7A2E"])
TERR = LinearSegmentedColormap.from_list("t", ["#0B0E0F", "#14201C", "#1E3128"])
plt.rcParams.update({"font.family": "monospace", "text.color": INK})
END = {1: "obstacle", 2: "terrain", 3: "zone", 4: "time", 5: "finish", 6: "margin"}


def canvas(title):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    o, s = T["terrain"]["origin"], T["terrain"]["size"]
    ax.imshow(H, origin="lower", extent=[o[0], o[0] + s[0], o[2], o[2] + s[2]], cmap=TERR, alpha=1)
    ax.contour(np.linspace(o[0], o[0] + s[0], H.shape[1]), np.linspace(o[2], o[2] + s[2], H.shape[0]), H,
               levels=np.arange(1, 12, 1.5), colors="#2A3A34", linewidths=.6)
    for k in T["capsules"]:
        a, b = k["a"], k["b"]
        if abs(a[1] - b[1]) > 1: ax.add_patch(plt.Circle((a[0], a[2]), max(k["r"], .4), color="#3B4A42", zorder=2))
        else: ax.plot([a[0], b[0]], [a[2], b[2]], color="#5A5236", lw=4, solid_capstyle="round", zorder=2)
    for e in T["spheres"]: ax.add_patch(plt.Circle((e["c"][0], e["c"][2]), e["r"], color="#24352C", alpha=.9, zorder=2))
    for g in T["gates"]:
        c, r = np.array(g["center"]), np.array(g["right"]) * g["half_width"] * 1.7
        ax.plot([c[0] - r[0], c[0] + r[0]], [c[2] - r[2], c[2] + r[2]], color=INK, lw=2.6, solid_capstyle="round", zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-185, 185); ax.set_ylim(-140, 120)
    fig.text(.04, .94, title, fontsize=20, fontweight="bold", color=INK)
    return fig, ax


def speed_line(ax, P, s, vmax=35.0, lw=2.6, alpha=1.0):
    seg = np.stack([P[:-1, [0, 2]], P[1:, [0, 2]]], axis=1)
    lc = LineCollection(seg, cmap=SPEED, norm=plt.Normalize(0, vmax), linewidths=lw, alpha=alpha, zorder=4, capstyle="round")
    lc.set_array(s[:-1]); ax.add_collection(lc)


def save(fig, name):
    os.makedirs(MEDIA, exist_ok=True); fig.savefig(os.path.join(MEDIA, name), facecolor=BG); plt.close(fig); print("->", name)


def fly(ckpt, seconds=80, randomize=False):
    import torch, ppo, fpv_sim
    ck = torch.load(ckpt, map_location="cpu"); D = ck["norm_m"].shape[0]
    ag = ppo.Agent(D); ag.load_state_dict(ck["agent"]); ag.eval()
    nm = ppo.RunningNorm(D, "cpu"); nm.m, nm.v = ck["norm_m"], ck["norm_v"]
    E = fpv_sim.Racing(TRACK, 1, 7, {"randomize": randomize, "delay": 2, "p_start": 1.0, "bat_fixed": 1.0, "race": True,
                                     "max_steps": 10 ** 6, "gate_margin": 0.3})
    E.reset(); X, A = [E.state(0).copy()], []
    with torch.no_grad():
        for _ in range(seconds * 50):
            a = ag.mean(nm(torch.as_tensor(E.obs()))).clamp(-1, 1)[0].numpy().astype(float)
            _, te, tr = E.step_one(0, a); X.append(E.state(0).copy()); A.append(a)
            if te or tr: break
    return np.array(X), np.array(A), E.info(0), ck


def track():
    fig, ax = canvas("HILLSIDE · the synthetic track")
    L = np.array(T["line"]); ax.plot(L[:, 0], L[:, 2], color=AMBER, lw=1.2, ls=(0, (4, 4)), zorder=4)
    for g in T["gates"]:
        c = g["center"]; ax.text(c[0] + 5, c[2] + 5, f'{g["id"]:02d} {g["name"]}', fontsize=9, color=INK, zorder=6)
    s = T["start"]; ax.plot([s[0]], [s[2]], "s", color=AMBER, ms=8, zorder=6)
    lap = np.sum(np.linalg.norm(np.diff(L, axis=0), axis=1))
    fig.text(.04, .905, f"{len(T['gates'])} gates · {lap:.0f} m a lap · from {L[:, 1].min():.0f} to {L[:, 1].max():.0f} m of altitude · "
             f"{len(T['boxes'])} gate frames, {len(T['capsules'])} poles, bales and trunks, {len(T['spheres'])} tree crowns", fontsize=11, color=GREY)
    save(fig, "track.png")


def checkpoints(runs):
    out = []
    for r in runs:
        out += sorted(glob.glob(os.path.join(r, "eval_*.pt")))
    return out


def learning(runs):
    cks = checkpoints(runs)
    pick = [cks[int(i)] for i in np.linspace(0, len(cks) - 1, min(9, len(cks)))]
    best = os.path.join(runs[-1], "best.pt")       # end on the policy that is published, not on the last checkpoint
    if os.path.exists(best): pick[-1] = best
    fig, ax = canvas("HOW IT LEARNED · one race per checkpoint, from the start pad")
    cols = plt.cm.viridis(np.linspace(.15, .95, len(pick)))
    rows = []
    for k, (c, col) in enumerate(zip(pick, cols)):
        X, A, inf, ck = fly(c)
        last = k == len(pick) - 1
        ax.plot(X[:, 0], X[:, 2], color=AMBER if last else col, lw=3 if last else 1.4, alpha=1 if last else .75, zorder=4 + last)
        if inf["end"] in (1, 2, 3):
            ax.plot([X[-1, 0]], [X[-1, 2]], marker="x", ms=14, mew=3.5, color=RED, zorder=7)
        run = os.path.basename(os.path.dirname(c))
        rows.append(f"{run:>6} it {ck['iter']:4d} · {inf['gates']:2d} gates · {END.get(inf['end'], '?'):>8}"
                    + (f" · best lap {inf['best_lap']:.1f} s" if inf["laps"] else ""))
        print(rows[-1])
    fig.text(.04, .905, "earliest in dark blue, the last in amber · X = a crash", fontsize=11, color=GREY)
    save(fig, "learning.png")


def race(ckpt):
    import matplotlib.animation as anim
    X, A, inf, ck = fly(ckpt)
    s = np.linalg.norm(X[:, 3:6], axis=1)
    fig, ax = canvas("THE TRAINED POLICY · a 3-lap race, simulator")
    speed_line(ax, X[:, :3], s)
    laps = f"laps {inf['last_lap']:.2f} s (best {inf['best_lap']:.2f})" if inf["laps"] else ""
    fig.text(.04, .905, f"{inf['gates']} gates · {END.get(inf['end'], '?')} · race {inf['race_time']:.2f} s · {laps} · top speed {s.max()*3.6:.0f} km/h · battery left {X[-1, 13]*100:.0f} %",
             fontsize=11, color=GREY)
    save(fig, "race.png")
    # telemetry strip
    t = np.arange(len(X)) * 0.02
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), dpi=110, sharex=True); fig.patch.set_facecolor(BG)
    for a_, y, lab, col in [(axs[0], s * 3.6, "speed km/h", AMBER), (axs[1], (A[:, 0] + 1) / 2 * 100, "throttle %", "#3FA39B"),
                            (axs[2], X[:, 13] * 100, "battery %", INK)]:
        a_.set_facecolor(BG); a_.plot(t[:len(y)], y, color=col, lw=1.4); a_.set_ylabel(lab, color=GREY)
        a_.tick_params(colors=GREY); [sp.set_color("#2A3034") for sp in a_.spines.values()]
    axs[2].set_xlabel("seconds", color=GREY); fig.tight_layout(); save(fig, "race-telemetry.png")
    # animated GIF, every 12th physics sample (12 fps -> 6x real time)
    fig, ax = canvas("THE TRAINED POLICY · simulator, 6x real time")
    L = np.array(T["line"]); ax.plot(L[:, 0], L[:, 2], color="#2A3A34", lw=1, zorder=3)
    trail, = ax.plot([], [], color=AMBER, lw=2.6, zorder=6); dot, = ax.plot([], [], "o", color=AMBER, ms=9, zorder=7)
    txt = fig.text(.04, .905, "", fontsize=13, color=INK)
    idx = list(range(0, len(X), 12))
    def upd(i):
        j = idx[i]; a0 = max(0, j - 150)
        trail.set_data(X[a0:j + 1, 0], X[a0:j + 1, 2]); dot.set_data([X[j, 0]], [X[j, 2]])
        txt.set_text(f"t {j*0.02:5.1f} s   {s[j]*3.6:4.0f} km/h   battery {X[j, 13]*100:3.0f} %")
        return trail, dot, txt
    a = anim.FuncAnimation(fig, upd, frames=len(idx), interval=80, blit=False)
    a.save(os.path.join(MEDIA, "race.gif"), writer=anim.PillowWriter(fps=12), dpi=46, savefig_kwargs={"facecolor": BG})
    plt.close(fig); print("-> race.gif")


def curve(runs):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    off = 0
    for r in runs:
        rows = [x for x in csv.DictReader(open(os.path.join(r, "log.csv"))) if x["ev_best_lap"] != ""]
        st = np.array([float(x["steps"]) for x in rows]) / 1e6 + off
        bl = np.array([float(x["ev_best_lap"]) if x["ev_best_lap"] not in ("", "nan") else np.nan for x in rows])
        rf = np.array([float(x["ev_race_frac"]) for x in rows])
        ax.plot(st, bl, color=AMBER, lw=2); ax.scatter(st[rf > 0], bl[rf > 0], color=INK, s=14, zorder=5)
        ax.axvline(off, color="#2A3034", lw=1); ax.text(off + 3, np.nanmax(bl) if np.isfinite(bl).any() else 60, os.path.basename(r), color=GREY)
        off = st[-1] if len(st) else off
    ax.set_xlabel("million decisions", color=GREY); ax.set_ylabel("best lap in evaluation (s)", color=GREY)
    ax.tick_params(colors=GREY); [sp.set_color("#2A3034") for sp in ax.spines.values()]
    fig.text(.06, .92, "TRAINING CURVE · white dots: the 3-lap race was finished", fontsize=14, color=INK, fontweight="bold")
    save(fig, "curve.png")


if __name__ == "__main__":
    cmd, args = sys.argv[1], [os.path.join(ROOT, a) if not os.path.isabs(a) else a for a in sys.argv[2:]]
    {"track": lambda: track(), "learning": lambda: learning(args), "race": lambda: race(args[0]), "curve": lambda: curve(args)}[cmd]()
