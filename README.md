# Embodied fly: dopamine → decision neurons → walking to food

A FlyWire v783 whole-brain spiking model (Shiu et al. 2024 parameters, 138,639 neurons, 15.1M connections) runs in closed loop with a NeuroMechFly body (flygym-gymnasium 1.3.2, MuJoCo physics) in an odor arena. You set neuromodulator levels and see how the fly's behavior changes.

```
odor at antennae ─► food / danger ORNs ─► whole brain ─► mushroom body ─► approach vs avoid MBONs = VALENCE
                                                           ▲
            dopamine knobs ─► real PAM / PPL1 neurons ─────┘ (scale KC→MBON synapses in their compartments)

VALENCE × food-odor L/R contrast ─► steering reflex ─► [left, right] leg drive ─► body walks
```

## Run

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python fetch_data.py                      # ~190 MB download, builds data/circuit.npz
.venv/Scripts/python test_brain.py                      # Brian2 fixture + dopamine locality
.venv/Scripts/python run.py --seed 0 --video base.mp4
.venv/Scripts/python run.py --seed 0 --dopamine-reward -1 --video no_reward_da.mp4
.venv/Scripts/python sweep.py --set knobs --seeds 10    # background job, ~1-2 h
```

Knobs range from −1 to +1:
- **+1** drives that neuron population with 50 Hz Poisson input.
- **−1** blocks its release.
- `--dopamine-reward` controls the PAM cluster.
- `--dopamine-punish` controls PPL1. Blocking it is hunger-like (Tsao 2018).
- `--octopamine` controls OA-VPM4.

Speed on a Ryzen 5 5600H is about 80 s of wall time per simulated second (body ≈ 37 s, brain ≈ 20–40 s). A 6 s trial takes about 8 minutes.

## 3D sandbox

```bash
.venv/Scripts/python export_assets.py         # once: fly meshes, walk clip, motion calibration, neuron positions
.venv/Scripts/python precompute_valence.py    # once (~2 h): the real brain's decision for 100 knob/odor settings
.venv/Scripts/python sandbox_server.py        # then open http://127.0.0.1:8765
```

- **Live preview (instant):**
  - Drag the food and danger around, and move the dopamine, octopamine, danger and heading sliders.
  - The decision uses the precomputed spiking-brain table, interpolated.
  - The body is the real NeuroMechFly mesh playing a recorded gait clip, moved by a kinematic model fitted to the physics body. The steering reflex is the same as `run.py`.
  - The brain view samples spikes from the real brain's firing rates at the nearest grid setting.
- **Run real simulation (~80 s of wall time per simulated second):**
  - Runs `run.py --record` with exactly your setup: full brain and MuJoCo physics.
  - The result then replays in the same scene with the recorded body poses (every 10 ms) and every recorded spike.
  - To reopen a past run, use `http://127.0.0.1:8765/?run=<id>`. The run folders are in `results/sandbox/`.
- **Live vs. real check:** the main-experiment arena, the same 8 start headings, and 10 wobble seeds per heading (`window.sandboxDebug.run` in the page).

  | Condition | Real | Live |
  |---|---|---|
  | baseline | 3/8 | 17% |
  | reward DA +1 | 8/8 | 100% |
  | reward DA −1 | 0/8 | 0% |
  | punish DA +1 | 1/8 | 10% |
  | punish DA −1 | 8/8 | 100% |
  | octopamine +1 | 5/8 | 33% |

  The main effects match. The preview is more cautious in the borderline conditions, so confirm those with a real run. The live walker has a hand-set heading wobble (0.35 rad/√s) standing in for gait sway.
- **Brain view:** the real FlyWire positions of all 138,639 neurons. The key circuit groups are coloured, and the strip at the bottom shows the chain: smell → Kenyon cells → approach/avoid MBONs (weakened by PPL1/PAM dopamine) → decision.

## What is data vs. what is assumed

| Part | Source |
|---|---|
| Neurons, synapse counts, signs, LIF parameters | Shiu et al. 2024 v783 files (`Connectivity_783.parquet`) |
| Neuron identities (ORN glomeruli, KC, MBON, PAM/PPL1, DNs) | FlyWire annotations (`Supplemental_file1_neuron_annotations.tsv`) |
| Which DAN type modulates which MBON (compartments) | **Derived from the connectome**: DAN→MBON synapse share ≥20% (`fetch_data.py` prints the table; it matches the literature, e.g. PPL101→MBON11 86%, PPL106→MBON14 95%, PAM11→MBON07 95%) |
| DAN/OA neurons have no fast synapses; they act only by scaling KC→MBON transmission, as `1/(1 + Σ share·rate/10 Hz)` | **Assumption.** No connectome-scale receptor map exists. |
| Approach MBONs (11, 12, 13, 14) vs avoidance MBONs (01, 03, 04, 26, 27) | Literature (Aso 2014, Tsao 2018) plus a path audit (MBON26/27 → MDN) |
| Steering reflex toward the stronger food odor, scaled by valence; innate turn away from danger | **Assumption / reflex.** See the finding below. |
| Octopamine increases walking speed | **Assumption** (readout gain) |
| Sensory ORN synapses forced excitatory | FlyWire transmitter predictions for ORNs are unreliable ("serotonin") |

## Findings so far

1. **The unmodified whole brain cannot steer.**
   - Stimulating left-only, right-only or both-side food ORNs (10–150 Hz, 3 seeds) drives about 150 descending neurons.
   - None of them carries a reliable left/right difference. DNa02_L fires about 50 Hz regardless of side, and DNa02_R about 0 Hz.
   - DNp09 (forward) and MDN (backward) stay silent.
   - So steering uses a labelled reflex. Other embodied projects hand-tune it too: NeuroFly steers mostly from the odor gradient, and Eon Systems says its gains were "chosen by hand".
2. **Dopamine does move the brain's food decision.** Brain-only runs (food odor both sides at 40 Hz, 6 seeds, mean approach − avoid MBON rate):

   | Condition | Approach Hz | Avoid Hz | Index |
   |---|---|---|---|
   | baseline | 31.7 | 14.5 | +17 |
   | reward DA +1 (PAM driven) | 29.1 | 0.0 | +29 |
   | reward DA −1 (PAM blocked) | 30.0 | 26.0 | +4 |
   | punish DA +1 (PPL1 driven) | 30.8 | 14.7 | +16 |
   | punish DA −1 (PPL1 blocked, hunger-like) | 132 | 14.2 | **+118** |
   | octopamine ±1 | 31.8 | 14.7 | +17 (no effect) |

3. **Closed-loop calibration passes.** The steering gain is frozen at flygym's 500 × valence, and the danger reflex gain is frozen at 80 × danger strength. Results over 8 seeds per condition, 6 s trials, food 30 mm away, random start heading over 360°:

   | Condition | Reached food | Mean closest approach |
   |---|---|---|
   | food odor | 8/8 | 1.9 mm |
   | odor off | 0/8 | 20 mm |
   | brain's food ORNs silenced (the reflex still smells the food) | 0/8 | 20 mm |

   The fly only uses the odor gradient when the brain's MBONs produce approach valence. Plot: `results/calibration_trajectories.png`.

4. **Main experiment: dopamine changes whether the fly passes the danger to reach food.**
   - Setup: food at (30, 0), danger strength 1.6 at (15, 0) on the direct path, 8 seeds, headings matched across conditions (`sweep.py --set knobs --danger 1.6`).
   - Plot: `results/knobs/knobs_trajectories.png`.

   | Condition | Mean valence | Reached | Mean time | Fisher p vs baseline |
   |---|---|---|---|---|
   | baseline | +0.39 | 3/8 | 4.8 s | – |
   | reward DA +1 | +0.92 | 8/8 | 3.2 s | 0.026 |
   | punish DA −1 (hunger-like) | +0.78 | 8/8 | 3.0 s | 0.026 |
   | reward DA −1 | +0.10 | 0/8 | – | 0.20 |
   | punish DA +1 | +0.37 | 1/8 | 5.9 s | 0.57 |
   | octopamine +1 | +0.39 | 5/8 | 4.0 s | 0.62 |

   - The p-values are uncorrected. Across 5 comparisons, Bonferroni needs p < 0.01, so the two 8/8 results are strong trends at n = 8. More seeds are needed.
   - The behavioral effect follows the brain's valence change by construction: the readout scales steering by valence. The non-trivial part is which knobs change valence, and by how much, in the spiking connectome.

5. **Model limitation.** With food odor, the brain's own PPL1 neurons fire at 170–370 Hz, which is unphysiological for this LIF model. That endogenous "punishment" dopamine already silences MBON11. This is why driving PPL1 further does nothing, while blocking it has the largest effect.

## Files

- `fetch_data.py`: download the data and build the groups and compartments.
- `brain.py`: numba LIF plus the modulation channel.
- `run.py`: one closed-loop trial.
- `sweep.py`: conditions × seeds, with statistics and a plot.
- `test_brain.py`: checks.
