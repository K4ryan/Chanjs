"""Conditions x seeds in parallel; prints reach-rate table and writes trajectories.png.
Example: python sweep.py --seeds 20 --danger 1.0 --set knobs"""
import argparse, json, multiprocessing as mp, pathlib
import numpy as np
from scipy.stats import fisher_exact

SETS = {
    "calibration": {  # step 4: does the brain's food valence make the fly find food?
        "odor": {},
        "odor_off": {"food_off": True},
        "food_orns_silenced": {"controls": ("food_orns_silenced",)},   # reflex still smells; brain doesn't
    },
    "baseline": {"baseline": {}},   # step 5: tune --danger until baseline reaches food ~50%
    "knobs": {  # step 6; -1 blocks that modulator's release, +1 drives it at 50 Hz
        "baseline": {},
        "reward_DA_up": {"dopamine_reward": 1.0},
        "reward_DA_blocked": {"dopamine_reward": -1.0},
        "punish_DA_up": {"dopamine_punish": 1.0},
        "punish_DA_blocked": {"dopamine_punish": -1.0},   # hunger-like (Tsao 2018)
        "octopamine_up": {"octopamine": 1.0},
    },
}


def job(args):
    name, cond, seed, seconds, danger, video = args
    import run
    cond = dict(cond)
    if cond.pop("food_off", False):
        run.FOOD_OFF = True
    r = run.trial(seed=seed, seconds=seconds, danger=danger, video=video, **cond)
    r["condition"] = name
    return r


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="knobs", choices=SETS)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--danger", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
    conds = SETS[a.set]
    jobs = [(n, c, s, a.seconds, a.danger, str(out / f"{a.set}_{n}.mp4") if s == 0 else None)
            for s in range(a.seeds) for n, c in conds.items()]
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        results = []
        for r in pool.imap_unordered(job, jobs):
            results.append(r)
            print(f"  {r['condition']:24s} seed {r['seed']:2d} reached={r['reached']} "
                  f"min_food={r['min_food_dist']:.1f}mm wall={r['wall_s']}s", flush=True)
    (out / f"{a.set}.json").write_text(json.dumps(results))
    base = next(iter(conds))
    kb = sum(r["reached"] for r in results if r["condition"] == base)
    print(f"\n{'condition':24s} reached     95% CI      p vs {base}  min food dist  path mm")
    for n in conds:
        rs = [r for r in results if r["condition"] == n]
        k, m = sum(r["reached"] for r in rs), len(rs)
        lo, hi = wilson(k, m)
        pv = fisher_exact([[k, m - k], [kb, m - kb]])[1] if n != base else float("nan")
        print(f"{n:24s} {k:2d}/{m:<2d}  [{lo:.2f}-{hi:.2f}]   {pv:8.3f}   "
              f"{np.mean([r['min_food_dist'] for r in rs]):6.1f}    {np.mean([r['path_mm'] for r in rs]):6.1f}")
    plot(results, conds, out / f"{a.set}_trajectories.png")


def plot(results, conds, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import run
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.4), sharex=True, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), conds):
        for r in (r for r in results if r["condition"] == n):
            xy = np.array([[q["x"], q["y"]] for q in r["rows"]])
            ax.plot(xy[:, 0], xy[:, 1], lw=.8, color="#2a7" if r["reached"] else "#999")
        ax.add_patch(plt.Circle(run.FOOD, 2, color="#f80", alpha=.5))
        ax.add_patch(plt.Circle(run.DANGER, 1, color="#27b", alpha=.5))
        ax.set_title(n, fontsize=9); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
