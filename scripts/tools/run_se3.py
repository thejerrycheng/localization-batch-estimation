#!/usr/bin/env python3
"""
The Assignment 3 experiment: SE(3) batch estimation against a sliding window,
with a stereo camera and an IMU.

    python scripts/tools/run_se3.py --seeds 3

Figures:
    se3_trajectory.pdf   the 3-D path, the landmarks, dead reckoning, estimates
    se3_window.pdf       translation and rotation error, cost, consistency
    se3_blackout.pdf     error against time, with the visible-landmark count
    se3_results.json     every number
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from se3 import SE3Problem, run, errors            # noqa: E402

INK, HI, GOLD, GREEN, ASH = "#151820", "#E4442A", "#D9A13F", "#2E9E5B", "#7A7466"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.9,
                     "figure.dpi": 140, "savefig.bbox": "tight", "legend.frameon": False})
WINDOWS = [1, 2, 5, 10, 20]
KW = dict(T=160, n_land=40, blackout=(80, 100))


def positions(Ts):
    return np.array([np.linalg.inv(T)[:3, 3] for T in Ts])


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=170)
    plt.close(fig); print("  wrote", name)


def fig_trajectory(out):
    p = SE3Problem(seed=1, **KW)
    dr = positions(p.dead_reckon())
    gt = positions(p.T_true)
    est_b = positions(run(p, window=None)[0])
    est_f = positions(run(p, window=1)[0])
    fig = plt.figure(figsize=(7.4, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*p.land.T, s=16, marker="*", color=GOLD, alpha=.8, label="landmarks")
    ax.plot(*gt.T, color=INK, lw=2.4, label="ground truth")
    ax.plot(*dr.T, color=ASH, lw=1.3, ls="--", label="dead reckoning")
    ax.plot(*est_f.T, color=HI, lw=1.3, label="window = 1")
    ax.plot(*est_b.T, color=GREEN, lw=1.6, label="batch")
    b0, b1 = KW["blackout"]
    ax.plot(*gt[b0:b1].T, color=HI, lw=5, alpha=.28)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Assignment 3: a 6-DoF path, a stereo camera and 40 landmarks")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout(); save(fig, out, "se3_trajectory")


def fig_window(seeds, out):
    rows = []
    for w in WINDOWS + [None]:
        tr, ro, secs, ins = [], [], [], []
        for s in range(seeds):
            p = SE3Problem(seed=s, **KW)
            t0 = time.perf_counter()
            est, cov = run(p, window=w)
            secs.append(time.perf_counter() - t0)
            e = errors(est, p.T_true)
            tr.append(float(np.sqrt((e[:, :3] ** 2).sum(1).mean())))
            ro.append(float(np.sqrt((e[:, 3:] ** 2).sum(1).mean())))
            if cov is not None:
                sd = np.sqrt(np.maximum(np.array([np.diag(c) for c in cov]), 1e-14))
                ok = (np.abs(e[1:]) <= 3 * sd[1:]).all(axis=1)
                ins.append(float(ok.mean()))
        rows.append(dict(window=("batch" if w is None else w),
                         trans=float(np.mean(tr)), trans_sd=float(np.std(tr)),
                         rot=float(np.mean(ro)), seconds=float(np.mean(secs)),
                         inside_3sigma=(float(np.mean(ins)) if ins else None)))
        print(f"  window {str(rows[-1]['window']):>5}  trans {np.mean(tr):.4f} m  "
              f"rot {np.mean(ro):.5f} rad  {np.mean(secs):5.1f} s"
              + (f"  inside 3σ {np.mean(ins)*100:5.1f}%" if ins else ""))

    fin = [r for r in rows if r["window"] != "batch"]
    bat = [r for r in rows if r["window"] == "batch"][0]
    x = [r["window"] for r in fin]
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 2.9))
    ax[0].errorbar(x, [r["trans"] for r in fin], yerr=[r["trans_sd"] for r in fin],
                   fmt="o-", color=HI, lw=2, ms=5, capsize=3, label="sliding window")
    ax[0].axhline(bat["trans"], color=GREEN, ls="--", lw=1.6, label="batch")
    ax[0].set_ylabel("translation RMS  [m]"); ax[0].legend(fontsize=8)
    ax[1].plot(x, [r["rot"] * 1000 for r in fin], "s-", color=GOLD, lw=2, ms=5)
    ax[1].axhline(bat["rot"] * 1000, color=GREEN, ls="--", lw=1.6)
    ax[1].set_ylabel("rotation RMS  [mrad]")
    ax[2].plot(x, [r["seconds"] for r in fin], "^-", color=INK, lw=2, ms=6)
    ax[2].axhline(bat["seconds"], color=GREEN, ls="--", lw=1.6)
    ax[2].set_ylabel(f"time for {KW['T']} steps  [s]"); ax[2].set_yscale("log")
    for a, t in zip(ax, ["translation", "rotation", "cost"]):
        a.set_xscale("log"); a.set_xlabel("window size [steps back]")
        a.set_title(t); a.grid(True, ls=":", alpha=.35)
    fig.tight_layout(); save(fig, out, "se3_window")
    return rows


def fig_blackout(out):
    p = SE3Problem(seed=1, **KW)
    gt = positions(p.T_true)
    runs = [("window = 1", run(p, window=1)[0], HI),
            ("window = 10", run(p, window=10)[0], GOLD),
            ("batch", run(p, window=None)[0], GREEN)]
    t = np.arange(p.T + 1) * p.dt
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})
    ax[0].fill_between(t, 0, p.visible, step="mid", color=GOLD, alpha=.55, lw=0)
    ax[0].set_ylabel("landmarks\nin view"); ax[0].grid(True, ls=":", alpha=.35)
    for lab, Ts, col in runs:
        d = np.linalg.norm(positions(Ts) - gt, axis=1)
        ax[1].plot(t, d, color=col, lw=1.5, label=lab)
    dr = np.linalg.norm(positions(p.dead_reckon()) - gt, axis=1)
    ax[1].plot(t, dr, color=ASH, lw=1.2, ls="--", label="dead reckoning")
    ax[1].set_xlabel("time  [s]"); ax[1].set_ylabel("position error  [m]")
    ax[1].legend(fontsize=8, ncol=4); ax[1].grid(True, ls=":", alpha=.35)
    ax[0].set_title("Uncertainty is driven by how many landmarks are in view")
    fig.tight_layout(); save(fig, out, "se3_blackout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    print("trajectory:"); fig_trajectory(out)
    print(f"window sweep, {a.seeds} seeds:"); rows = fig_window(a.seeds, out)
    print("blackout:"); fig_blackout(out)
    with open(os.path.join(out, "se3_results.json"), "w") as f:
        json.dump(dict(seeds=a.seeds, params=KW, windows=rows), f, indent=2)
    print("  wrote se3_results.json")


if __name__ == "__main__":
    main()
