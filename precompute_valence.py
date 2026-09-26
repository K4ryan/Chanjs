"""Precompute the real spiking brain's decision for the sandbox live preview.

For each (reward DA, punish DA, food-odor ORN rate) grid point: run the whole brain 3 s with the
same knob/modulation code as run.py, measure the last 1 s. Saves sandbox/assets/valence.json
(approach/avoid MBON rates, valence, modulator traces) and brain_rates.bin (per-neuron firing
rates, sparse). Octopamine is not a grid axis: +-1 left MBON rates unchanged (README finding 2).
Run after data exists: python precompute_valence.py  (~35 min on 4 workers)"""
import itertools, json, multiprocessing as mp, pathlib
import numpy as np

OUT = pathlib.Path(__file__).parent / "sandbox" / "assets"
KNOB = [-1.0, -0.5, 0.0, 0.5, 1.0]
ODOR_HZ = [15.0, 40.0, 80.0, 130.0]      # food ORN rates seen from ~30 mm down to ~3 mm
WARM_MS, MEASURE_MS = 2000.0, 1000.0


def point(args):
    reward, punish, odor = args
    import run
    from brain import Brain
    b = Brain.load(seed=0)
    run.set_knob(b, b.pam, reward)
    run.set_knob(b, b.ppl1, punish)
    b.set_rate("food_L", odor); b.set_rate("food_R", odor)
    acc = np.zeros(b.n)
    for w in range(int((WARM_MS + MEASURE_MS) / run.WINDOW_MS)):
        c = b.run(run.WINDOW_MS)
        b.update_modulation(c, run.WINDOW_MS)
        if w * run.WINDOW_MS >= WARM_MS:
            acc += c
    rates = acc * 1000.0 / MEASURE_MS
    grp = lambda names: float(np.mean([rates[b.groups[m]].mean() for m in names]))
    a, v = grp(run.APPROACH), grp(run.AVOID)
    idx = np.nonzero(rates)[0].astype(np.uint32)
    return {"reward": reward, "punish": punish, "odor_hz": odor, "approach": a, "avoid": v,
            "valence": (a - v) / (a + v + 1.0), "oa_hz": float(b.trace["OA_VPM4"]),
            "traces": {k: round(float(t), 2) for k, t in b.trace.items()},
            "mbon": {m: round(grp([m]), 2) for m in run.APPROACH + run.AVOID},
            "_idx": idx, "_rates": rates[idx].astype(np.float32)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(KNOB, KNOB, ODOR_HZ))
    with mp.Pool(4, maxtasksperchild=1) as pool:
        results = []
        for r in pool.imap(point, grid):
            results.append(r)
            print(f"reward {r['reward']:+.1f} punish {r['punish']:+.1f} odor {r['odor_hz']:5.0f} Hz -> "
                  f"approach {r['approach']:6.1f} avoid {r['avoid']:6.1f} valence {r['valence']:+.2f} "
                  f"({len(r['_idx'])} active)", flush=True)
    idx_all, rate_all, off = [], [], 0
    for r in results:
        n = len(r["_idx"])
        r["sparse"] = [off, n]; off += n
        idx_all.append(r.pop("_idx")); rate_all.append(r.pop("_rates"))
    np.concatenate(idx_all).tofile(OUT / "brain_idx.bin")
    np.concatenate(rate_all).tofile(OUT / "brain_rates.bin")
    (OUT / "valence.json").write_text(json.dumps({"knob": KNOB, "odor_hz": ODOR_HZ, "points": results}))
    print("wrote", OUT / "valence.json")


if __name__ == "__main__":
    main()
