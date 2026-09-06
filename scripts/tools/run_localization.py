#!/usr/bin/env python3
"""
Batch vs sliding-window localisation, quantified.

    python scripts/tools/run_localization.py --seeds 5

Figures (vector PDF + PNG into docs/figures/):
    loc_trajectory.pdf   the map, the dead-reckoned path, and the estimates
    loc_window.pdf       error, runtime and consistency against window size
    loc_blackout.pdf     what happens when the landmarks disappear
    loc_results.json     every number
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimator import Problem, run, errors            # noqa: E402

INK, HI, GOLD, GREEN, ASH = "#151820", "#E4442A", "#D9A13F", "#2E9E5B", "#7A7466"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.9,
                     "figure.dpi": 140, "savefig.bbox": "tight", "legend.frameon": False})
WINDOWS = [1, 2, 5, 10, 20, 50]
KW = dict(T=300, sigma_v=0.15, sigma_w=0.06)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=170)
    plt.close(fig); print("  wrote", name)


def fig_trajectory(out):
    p = Problem(seed=1, **KW)
    dr = p.dead_reckon()
    batch, _ = run(p, window=None)
    ekf, cov = run(p, window=1)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(p.land[:, 0], p.land[:, 1], s=70, marker="*", color=GOLD,
               edgecolor=INK, lw=0.6, zorder=5, label="landmarks")
    ax.plot(p.x_true[:, 0], p.x_true[:, 1], color=INK, lw=2.4, label="ground truth")
    ax.plot(dr[:, 0], dr[:, 1], color=ASH, lw=1.4, ls="--", label="dead reckoning")
    ax.plot(ekf[:, 0], ekf[:, 1], color=HI, lw=1.4, label="window = 1 (a filter)")
    ax.plot(batch[:, 0], batch[:, 1], color=GREEN, lw=1.6, label="batch")
    b0, b1 = 150, 200
    ax.plot(p.x_true[b0:b1, 0], p.x_true[b0:b1, 1], color=HI, lw=5, alpha=0.25,
            solid_capstyle="round", zorder=1)
    ax.annotate("no landmarks visible", p.x_true[(b0 + b1) // 2, :2],
                textcoords="offset points", xytext=(10, 14), fontsize=8, color=HI)
    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]"); ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, ls=":", alpha=.35)
    ax.set_title("Lost in the Woods: 300 steps, 17 landmarks, a 50-step blackout")
    fig.tight_layout(); save(fig, out, "loc_trajectory")


def fig_window(seeds, out):
    rows = []
    for w in WINDOWS + [None]:
        rms, rmsth, secs, inside = [], [], [], []
        for s in range(seeds):
            p = Problem(seed=s, **KW)
            t0 = time.perf_counter()
            est, cov = run(p, window=w)
            secs.append(time.perf_counter() - t0)
            e = errors(est, p.x_true)
            rms.append(float(np.sqrt((e[:, :2] ** 2).sum(1).mean())))
            rmsth.append(float(np.sqrt((e[:, 2] ** 2).mean())))
            if cov is not None:
                # consistency: how often the error sits inside its own 3-sigma
                sd = np.sqrt(np.maximum(np.einsum("kii->ki", cov), 1e-12))
                ok = (np.abs(e[1:]) <= 3 * sd[1:]).all(axis=1)
                inside.append(float(ok.mean()))
        rows.append(dict(window=("batch" if w is None else w),
                         rms_xy=float(np.mean(rms)), rms_xy_sd=float(np.std(rms)),
                         rms_theta=float(np.mean(rmsth)),
                         seconds=float(np.mean(secs)),
                         inside_3sigma=(float(np.mean(inside)) if inside else None)))
        print(f"  window {str(rows[-1]['window']):>5}  rms xy {np.mean(rms):.4f} m  "
              f"rms th {np.mean(rmsth):.4f} rad  {np.mean(secs):5.2f} s"
              + (f"  inside 3σ {np.mean(inside)*100:5.1f}%" if inside else ""))

    fin = [r for r in rows if r["window"] != "batch"]
    bat = [r for r in rows if r["window"] == "batch"][0]
    x = [r["window"] for r in fin]
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 2.9))
    ax[0].errorbar(x, [r["rms_xy"] for r in fin], yerr=[r["rms_xy_sd"] for r in fin],
                   fmt="o-", color=HI, lw=2, ms=5, capsize=3, label="sliding window")
    ax[0].axhline(bat["rms_xy"], color=GREEN, ls="--", lw=1.6, label="batch")
    ax[0].set_xscale("log"); ax[0].set_xlabel("window size [steps back]")
    ax[0].set_ylabel("position RMS error  [m]"); ax[0].legend(fontsize=8)
    ax[0].set_title("accuracy")

    ax[1].plot(x, [r["seconds"] for r in fin], "s-", color=GOLD, lw=2, ms=5)
    ax[1].axhline(bat["seconds"], color=GREEN, ls="--", lw=1.6)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("window size [steps back]"); ax[1].set_ylabel("time for 300 steps  [s]")
    ax[1].set_title("cost")

    ins = [r["inside_3sigma"] * 100 for r in fin]
    ax[2].plot(x, ins, "^-", color=INK, lw=2, ms=6)
    ax[2].axhline(99.7, color=GREEN, ls="--", lw=1.6)
    ax[2].text(x[0], 100.6, "what 3σ should cover", fontsize=7.5, color=GREEN)
    ax[2].set_xscale("log"); ax[2].set_xlabel("window size [steps back]")
    ax[2].set_ylabel("errors inside their own 3σ  [%]")
    ax[2].set_ylim(0, 108); ax[2].set_title("consistency")
    for a in ax:
        a.grid(True, ls=":", alpha=.35)
    fig.tight_layout(); save(fig, out, "loc_window")
    return rows


def fig_mismatch(seeds, out):
    """Is a sliding window inherently over-confident, or only when the noise
    model is wrong?

    The report finds that the sliding-window 3-sigma bounds fail to cover the
    error while the batch bounds are 'perfectly bounded', and suspects a poor
    initial covariance estimate.  This tests that directly: keep the true noise
    fixed and scale the covariances the estimator *believes*.
    """
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for sc in scales:
        ins, rms = [], []
        for s in range(seeds):
            p = Problem(seed=s, **KW)
            p.Q = p.Q * sc
            p.R = p.R * sc
            est, cov = run(p, window=10)
            e = errors(est, p.x_true)
            sd = np.sqrt(np.maximum(np.einsum("kii->ki", cov), 1e-12))
            ok = (np.abs(e[1:]) <= 3 * sd[1:]).all(axis=1)
            ins.append(float(ok.mean()))
            rms.append(float(np.sqrt((e[:, :2] ** 2).sum(1).mean())))
        rows.append(dict(scale=sc, inside_3sigma=float(np.mean(ins)),
                         rms_xy=float(np.mean(rms))))
        print(f"  assumed covariance x{sc:<5g} inside 3σ {np.mean(ins)*100:5.1f}%   "
              f"rms {np.mean(rms):.4f} m")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = [r["scale"] for r in rows]
    ax.plot(x, [r["inside_3sigma"] * 100 for r in rows], "o-", color=HI, lw=2, ms=6)
    ax.axhline(99.7, color=GREEN, ls="--", lw=1.6)
    ax.text(x[0], 100.2, "what 3σ should cover", fontsize=7.5, color=GREEN)
    ax.axvline(1.0, color=INK, ls=":", lw=1.2)
    ax.text(1.05, 40, "the true noise", fontsize=7.5, color=INK)
    ax.set_xscale("log"); ax.set_xlabel("covariance the estimator assumes  (× the truth)")
    ax.set_ylabel("errors inside their own 3σ  [%]")
    ax.set_title(f"A window is over-confident only when the noise model is wrong ({seeds} seeds)")
    ax.set_ylim(0, 108); ax.grid(True, ls=":", alpha=.35)
    fig.tight_layout(); save(fig, out, "loc_mismatch")
    return rows


def fig_blackout(out):
    p = Problem(seed=1, **KW)
    est1, cov1 = run(p, window=1)
    est20, cov20 = run(p, window=20)
    batch, _ = run(p, window=None)
    e1, e20, eb = (errors(v, p.x_true) for v in (est1, est20, batch))
    t = np.arange(p.T + 1) * p.dt
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})
    ax[0].fill_between(t, 0, p.visible, step="mid", color=GOLD, alpha=.55, lw=0)
    ax[0].set_ylabel("landmarks\nvisible"); ax[0].grid(True, ls=":", alpha=.35)
    for e, lab, col in [(e1, "window = 1", HI), (e20, "window = 20", GOLD), (eb, "batch", GREEN)]:
        ax[1].plot(t, np.hypot(e[:, 0], e[:, 1]), color=col, lw=1.5, label=lab)
    sd = np.sqrt(np.maximum(np.einsum("kii->ki", cov1)[:, :2].sum(1), 1e-12))
    ax[1].plot(t, 3 * sd, color=INK, lw=1.1, ls=":", label="3σ, window = 1")
    ax[1].set_xlabel("time  [s]"); ax[1].set_ylabel("position error  [m]")
    ax[1].legend(fontsize=8, ncol=4); ax[1].grid(True, ls=":", alpha=.35)
    ax[0].set_title("Error tracks the landmark count, and the filter's 3σ does not keep up")
    fig.tight_layout(); save(fig, out, "loc_blackout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    print("trajectory figure:"); fig_trajectory(out)
    print(f"window sweep, {a.seeds} seeds:")
    rows = fig_window(a.seeds, out)
    print("covariance mismatch:")
    mism = fig_mismatch(a.seeds, out)
    print("blackout figure:"); fig_blackout(out)
    with open(os.path.join(out, "loc_results.json"), "w") as f:
        json.dump(dict(seeds=a.seeds, params=KW, windows=rows, mismatch=mism), f, indent=2)
    print("  wrote loc_results.json")


if __name__ == "__main__":
    main()
