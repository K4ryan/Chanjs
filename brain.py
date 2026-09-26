"""Whole-brain LIF (Shiu et al. 2024 parameters), a numba port of ../model.js FlyBrain,
plus a slow neuromodulatory channel:

  DAN / OA spikes -> per-type rate trace (tau 1 s)
  -> KC->MBON transmission in that DAN's compartments scaled by 1/(1 + sum_d f_dM * rate_d / D50)

The fast synapses of DAN/OA neurons are removed (Shiu treats them as ordinary excitatory
synapses). The compartment shares f_dM come from the connectome (fetch_data.py); the
depression rule and D50 are modelling assumptions, not measured receptor physiology."""
import json, pathlib
import numpy as np
from numba import njit

DATA = pathlib.Path(__file__).parent / "data"
DT = 0.1                      # ms
A, B = np.exp(-DT / 20), np.exp(-DT / 5)   # tau_m 20 ms, tau_g 5 ms
C = (A - B) / 3               # exact coupling term for tau_g/(tau_m - tau_g) = 1/3
REST, THRESH = -52.0, -45.0
REFRACTORY = 22               # ticks (2.2 ms)
DELAY = 18                    # ticks (1.8 ms)
W_SCALE = 0.275               # mV per synapse
KICK = 68.75                  # mV per Poisson input event
TRACE_TAU = 1000.0            # ms, neuromodulator trace
D50 = 10.0                    # Hz of DAN firing that halves KC->MBON transmission


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _run(ticks, tick0, v, g, last, ref, offsets, targets, w, silenced, prob, inputs,
         kick_idx, kick_amt, qbuf, qlen, counts, free, spk):
    n = v.shape[0]
    for t in range(ticks):
        tick = tick0 + t
        ns = 0
        for i in range(n):
            if tick - last[i] < ref[i]:
                free[i] = False
                continue
            free[i] = True
            if g[i] == 0.0 and v[i] == REST:
                continue          # exactly at rest: stays at rest, cannot spike
            old = g[i]
            g[i] = old * B
            v[i] = REST + (v[i] - REST) * A + old * C
            # ponytail: snap residues <1e-9 mV to rest so dormant cells are skipped (deviation far below any threshold effect)
            if abs(g[i]) < 1e-9 and abs(v[i] - REST) < 1e-9:
                g[i] = 0.0
                v[i] = REST
            if v[i] > THRESH:
                spk[ns] = i
                ns += 1
                last[i] = tick
                free[i] = False
        s = tick % 19
        for k in range(qlen[s]):
            src = qbuf[s, k]
            if silenced[src]:
                continue
            for e in range(offsets[src], offsets[src + 1]):
                tg = targets[e]
                if free[tg]:
                    g[tg] += w[e]
        qlen[s] = 0
        if t == 0:
            for k in range(kick_idx.shape[0]):
                if free[kick_idx[k]]:
                    v[kick_idx[k]] += kick_amt[k]
        for k in range(inputs.shape[0]):
            i = inputs[k]
            if free[i] and np.random.random() < prob[i]:
                v[i] += KICK
        f = (tick + DELAY) % 19
        for k in range(ns):
            i = spk[k]
            v[i] = REST
            g[i] = 0.0
            qbuf[f, k] = i
            counts[i] += 1
        qlen[f] = ns


class Brain:
    def __init__(self, offsets, targets, weights, seed=0, groups=None):
        n = len(offsets) - 1
        self.n, self.offsets, self.targets = n, np.asarray(offsets, np.int64), np.asarray(targets, np.int32)
        self.base_w = np.asarray(weights, np.float64) * W_SCALE
        self.w = self.base_w.copy()
        self.v = np.full(n, REST); self.g = np.zeros(n)
        self.last = np.full(n, -10**9, np.int64); self.ref = np.full(n, REFRACTORY, np.int64)
        self.silenced = np.zeros(n, np.bool_); self.prob = np.zeros(n)
        self.inputs = np.zeros(0, np.int64)
        self.qbuf = np.zeros((19, n), np.int32); self.qlen = np.zeros(19, np.int64)
        self.counts = np.zeros(n, np.int64); self.free = np.zeros(n, np.bool_); self.spk = np.zeros(n, np.int64)
        self.tick = 0
        self.groups = {k: np.asarray(v, np.int64) for k, v in (groups or {}).items()}
        _seed(seed)

    @classmethod
    def load(cls, seed=0):
        z = np.load(DATA / "circuit.npz")
        meta = json.loads((DATA / "groups.json").read_text())
        w = np.where(z["modulatory_pre"], 0, z["weights"])
        brain = cls(z["offsets"], z["targets"], w, seed, meta["groups"])
        edge, kind = z["kc_mbon_edge"], z["kc_mbon_type"]
        brain.comp = meta["compartments"]
        brain.pam, brain.ppl1 = meta["pam"], meta["ppl1"]
        brain.trace = {d: 0.0 for d in sorted({c["dan"] for c in brain.comp}) + ["OA_VPM4"]}
        brain.mbon_edges = {m: edge[kind == m] for m in set(c["mbon"] for c in brain.comp) | {"MBON11"}}
        brain.gate_modulation = True   # control: False keeps KC->MBON at baseline whatever the DANs do
        brain.mod_off = set()          # DAN/OA types whose release is blocked (silenced / negative knob)
        return brain

    def idx(self, group):
        return self.groups[group] if isinstance(group, str) else np.atleast_1d(np.asarray(group, np.int64))

    def set_rate(self, group, hz):
        i = self.idx(group)
        self.prob[i] = min(max(hz, 0.0), 200.0) * 1e-4
        self.ref[i] = 0   # stimulated neurons have rfc = 0 in Shiu et al.
        self.inputs = np.where(self.prob > 0)[0]

    def silence(self, group, on=True):
        self.silenced[self.idx(group)] = on

    def run(self, ms, kick=None):
        ticks = int(round(ms / DT))
        ki, ka = (np.zeros(0, np.int64), np.zeros(0)) if kick is None else (
            np.asarray([k for k, _ in kick], np.int64), np.asarray([a for _, a in kick], np.float64))
        self.counts[:] = 0
        _run(ticks, self.tick, self.v, self.g, self.last, self.ref, self.offsets, self.targets, self.w,
             self.silenced, self.prob, self.inputs, ki, ka, self.qbuf, self.qlen, self.counts, self.free, self.spk)
        self.tick += ticks
        return self.counts

    def rate(self, group, counts, ms):
        """Mean firing rate (Hz per neuron) of a group in the last window."""
        return counts[self.idx(group)].mean() * 1000.0 / ms

    def update_modulation(self, counts, ms):
        """Advance DAN/OA traces from this window's spikes and apply KC->MBON scaling."""
        k = 1 - np.exp(-ms / TRACE_TAU)
        for d in self.trace:
            self.trace[d] += k * ((0.0 if d in self.mod_off else self.rate(d, counts, ms)) - self.trace[d])
        if not self.gate_modulation:
            return
        load = {m: 0.0 for m in self.mbon_edges}
        for c in self.comp:
            load[c["mbon"]] += c["frac"] * self.trace[c["dan"]] / D50
        # ponytail: octopamine brake on MBON11 (Sayin 2019, OA-VPM4) uses the same rule; its gain is assumed.
        load["MBON11"] += self.trace["OA_VPM4"] / D50
        for m, e in self.mbon_edges.items():
            self.w[e] = self.base_w[e] / (1 + load[m])

    def multipliers(self):
        return {m: float(self.w[e].sum() / self.base_w[e].sum()) for m, e in self.mbon_edges.items() if len(e)}
