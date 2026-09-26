"""Run: python test_brain.py  (asserts; no framework)"""
import numpy as np
from brain import Brain


def network(n, edges):
    edges = sorted(edges)
    offsets = np.zeros(n + 1, np.int64)
    for s, _, _ in edges:
        offsets[s + 1] += 1
    return np.cumsum(offsets), [t for _, t, _ in edges], [w for _, _, w in edges]


def test_brian2_fixture():
    # Same fixture as ../tests/model.test.cjs (Brian2 2.10.1 reference spike times).
    b = Brain(*network(6, [(0, 2, 160), (1, 2, -100), (2, 3, 200), (3, 2, 60), (3, 4, 100), (4, 5, -100), (5, 4, 100)]))
    b.ref[[0, 1]] = 0
    sweet, bitter = {0, 1, 20, 40, 60, 80, 120, 200, 300, 500}, {40, 60, 81, 120, 210, 310}
    observed = []
    for tick in range(1000):
        kick = [(0, 68.75)] * (tick in sweet) + [(1, 68.75)] * (tick in bitter)
        counts = b.run(0.1, kick or None)
        observed += [(int(i), round(tick * .1, 7)) for i in np.where(counts)[0]]
    assert observed == [(0, .1), (0, 2.1), (0, 4.1), (1, 4.1), (2, 5.1), (0, 6.1), (1, 6.1), (0, 8.1), (1, 8.2), (3, 11.2),
                        (0, 12.1), (1, 12.1), (2, 15.1), (0, 20.1), (1, 21.1), (3, 21.2), (4, 26.1), (0, 30.1), (1, 31.1),
                        (2, 32.9), (3, 39), (0, 50.1), (2, 55), (3, 61.1)], observed


def test_dopamine_only_changes_kc_mbon():
    # Matched seeds: driving PPL101 must depress KC->MBON11 and touch no other synapse.
    a, b = Brain.load(seed=7), Brain.load(seed=7)
    b.set_rate("PPL101", 50)
    for _ in range(40):
        for x in (a, b):
            x.update_modulation(x.run(25), 25)
    changed = np.where(a.w != b.w)[0]
    allowed = np.concatenate([b.mbon_edges[c["mbon"]] for c in b.comp if c["dan"] == "PPL101"])
    assert len(changed) and set(changed) <= set(allowed)
    assert b.multipliers()["MBON11"] < 0.9 and a.multipliers()["MBON11"] == 1.0


if __name__ == "__main__":
    test_brian2_fixture(); print("fixture ok")
    test_dopamine_only_changes_kc_mbon(); print("dopamine ok")
