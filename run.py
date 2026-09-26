"""One closed-loop trial: FlyWire brain <-> NeuroMechFly body in an odor arena.

  antenna/palp odor -> Poisson rates on food / danger ORNs (per side)
  brain (25 ms)     -> mushroom-body output neurons: approach MBONs vs avoidance MBONs
                       = the fly's VALENCE for the food odor (where dopamine acts)
  body              -> a labelled steering REFLEX turns toward the stronger food odor, scaled
                       by that valence (negative valence turns away), and away from danger.

Why a reflex: the unmodified whole-brain LIF carries no reliable left/right odor information
in its descending neurons (README, "Findings"), so steering from DNs would be noise. The
decision (approach vs avoid, and how strongly) is the brain's; the geometry is the reflex's.

Example: python run.py --seed 0 --dopamine-reward 1 --danger 1 --video reward.mp4"""
import argparse, csv, json, pathlib, time
import numpy as np
from brain import Brain

WINDOW_MS = 25.0
BODY_DT = 2e-4                  # s; walks like the 1e-4 default at half the cost (checked in step 1)
ORN_MAX_HZ, ORN_HALF = 150.0, 0.01  # saturating odor -> ORN rate: 150 * I / (I + I_half)
KNOB_HZ = 50.0                  # Poisson drive to DAN / OA neurons at knob = +1; knob < 0 blocks their release
SMOOTH_MS = 100.0
APPROACH = ["MBON11", "MBON12", "MBON13", "MBON14"]            # needed for food seeking (Tsao 2018)
AVOID = ["MBON01", "MBON03", "MBON04", "MBON26", "MBON27"]     # avoidance / drive MDN (Aso 2014; path audit)

# Reflex + readout constants (calibrated once in step 4; see README).
G_FOOD = 500.0      # flygym example's attractive gain, multiplied by valence
G_DANGER = 80.0     # innate danger avoidance gain (flygym example default)
DRIVE = 1.0         # walking drive (flies walk spontaneously; an explicit assumption)
AROUSAL = 0.5       # ponytail: octopamine speeds walking via this assumed gain, not a traced circuit

FOOD = np.array([30.0, 0.0])
DANGER = np.array([15.0, 0.0])
FOOD_OFF = False    # control: food is there but emits no odor


def make_sim(danger, video, heading=0.0, food=FOOD, danger_pos=DANGER):
    from flygym_gymnasium import Fly, Camera
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.examples.locomotion import HybridTurningController
    src = [[*food, 1.5], [*danger_pos, 1.5]]
    peak = [[0.0 if FOOD_OFF else 1.0, 0.0], [0.0, danger]]
    colors = [[1, .5, .05, 1], [.12, .47, .71, 1 if danger > 0 else 0]]
    arena = OdorArena(odor_source=np.array(src), peak_odor_intensity=np.array(peak),
                      diffuse_func=lambda x: x ** -2, marker_colors=np.array(colors), marker_size=0.3)
    legs = [f"{l}{s}" for l in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for s in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]
    fly = Fly(spawn_pos=(0, 0, 0.2), spawn_orientation=(0, 0, heading), contact_sensor_placements=legs,
              enable_olfaction=True, enable_adhesion=True, draw_adhesion=False)
    cams = []
    if video:
        cams = [Camera(attachment_point=arena.root_element.worldbody, camera_name="birdeye_cam",
                       timestamp_text=True, camera_parameters={"mode": "fixed", "pos": (food[0] / 2, food[1] / 2, 30),
                                                               "euler": (0, 0, 0), "fovy": 45})]
    return HybridTurningController(fly=fly, cameras=cams, arena=arena, timestep=BODY_DT), cams


def lr(intensity_row):
    """[L, R] odor from [LPalp, RPalp, LAntenna, RAntenna], antenna-weighted as in flygym's example."""
    return np.average(np.asarray(intensity_row).reshape(2, 2), axis=0, weights=(1, 9))


def contrast(pair):
    m = pair.mean()
    return 0.0 if m <= 0 else (pair[0] - pair[1]) / m


def set_knob(brain, types, level):
    for d in types:
        brain.set_rate(d, KNOB_HZ * max(level, 0.0))
    if level < 0:
        brain.mod_off |= set(types)


def trial(seed=0, seconds=6.0, dopamine_reward=0.0, dopamine_punish=0.0, octopamine=0.0, danger=0.0,
          heading=None, video=None, controls=(), log=None, food=None, danger_pos=None, record=None):
    """record: directory -> writes run.json, poses.bin (10 ms mesh-geom world poses), spikes_*.bin (sparse per window)."""
    heading = np.random.default_rng(seed).uniform(-np.pi, np.pi) if heading is None else heading
    food = FOOD if food is None else np.asarray(food, float)
    danger_pos = DANGER if danger_pos is None else np.asarray(danger_pos, float)
    brain = Brain.load(seed=seed)
    set_knob(brain, brain.pam, dopamine_reward)
    set_knob(brain, brain.ppl1, dopamine_punish)
    set_knob(brain, ["OA_VPM4"], octopamine)
    if "dans_silenced" in controls:          # knob drive present, but no dopamine/OA release at all
        brain.mod_off |= set(brain.trace)
    if "target_disabled" in controls:        # DANs fire, but KC->MBON synapses ignore them
        brain.gate_modulation = False
    if "food_orns_silenced" in controls:
        brain.silence("food_L"); brain.silence("food_R")

    sim, cams = make_sim(danger, video, heading, food, danger_pos)
    obs, _ = sim.reset(seed=seed)
    smooth = {"approach": 0.0, "avoid": 0.0}
    alpha = 1 - np.exp(-WINDOW_MS / SMOOTH_MS)
    rows, reached, t0 = [], None, time.time()
    steps = int(WINDOW_MS / 1000 / BODY_DT)
    if record:
        m = sim.physics.model
        geoms = [g for g in range(m.ngeom) if m.geom_type[g] == 7 and m.geom_dataid[g] >= 0]   # same order as export_assets
        frame_every = int(0.01 / BODY_DT)
        poses, sp_idx, sp_cnt, windows = [], [], [], []
        pam = [d for d in brain.trace if d.startswith("PAM")]
        ppl1 = [d for d in brain.trace if d.startswith("PPL1")]
    for w in range(int(seconds * 1000 / WINDOW_MS)):
        food_lr, danger_lr = lr(obs["odor_intensity"][0]), lr(obs["odor_intensity"][1])
        for side, i in (("L", 0), ("R", 1)):
            brain.set_rate(f"food_{side}", ORN_MAX_HZ * food_lr[i] / (food_lr[i] + ORN_HALF))
            brain.set_rate(f"danger_{side}", ORN_MAX_HZ * danger_lr[i] / (danger_lr[i] + ORN_HALF))
        counts = brain.run(WINDOW_MS)
        brain.update_modulation(counts, WINDOW_MS)
        smooth["approach"] += alpha * (np.mean([brain.rate(m, counts, WINDOW_MS) for m in APPROACH]) - smooth["approach"])
        smooth["avoid"] += alpha * (np.mean([brain.rate(m, counts, WINDOW_MS) for m in AVOID]) - smooth["avoid"])
        valence = (smooth["approach"] - smooth["avoid"]) / (smooth["approach"] + smooth["avoid"] + 1.0)
        # flygym's reflex form: negative bias -> slow the left legs -> turn left.
        bias = -G_FOOD * valence * contrast(food_lr) + G_DANGER * danger * contrast(danger_lr)
        turn = np.tanh(bias ** 2) * np.sign(bias)
        drive = DRIVE * (1 + AROUSAL * min(brain.trace["OA_VPM4"] / KNOB_HZ, 1.0))
        action = np.full(2, drive)
        action[int(turn > 0)] -= abs(turn) * 0.8 * drive
        for s in range(steps):
            if record and s % frame_every == 0:
                d = sim.physics.data
                poses.append(np.concatenate([d.geom_xpos[geoms], d.geom_xmat[geoms]], axis=1).astype(np.float32))
            obs, *_ = sim.step(action)
            if video:
                sim.render()
        xy = obs["fly"][0, :2]
        rows.append({"t": (w + 1) * WINDOW_MS / 1000, "x": float(xy[0]), "y": float(xy[1]),
                     "valence": round(valence, 3), "approach_hz": round(smooth["approach"], 1),
                     "avoid_hz": round(smooth["avoid"], 1), "dL": float(action[0]), "dR": float(action[1])})
        if record:
            nz = np.nonzero(counts)[0]
            windows.append([int(sum(len(a) for a in sp_idx)), int(len(nz))])
            sp_idx.append(nz.astype(np.uint32)); sp_cnt.append(np.minimum(counts[nz], 255).astype(np.uint8))
            rows[-1].update(orn_hz=round((brain.rate("food_L", counts, WINDOW_MS) + brain.rate("food_R", counts, WINDOW_MS)) / 2, 1),
                            kc_hz=round(brain.rate("KC", counts, WINDOW_MS), 2),
                            pam_hz=round(float(np.mean([brain.trace[d] for d in pam])), 1),
                            ppl1_hz=round(float(np.mean([brain.trace[d] for d in ppl1])), 1),
                            mult={m: round(v, 3) for m, v in brain.multipliers().items() if m in APPROACH + AVOID})
        if log:
            log(rows[-1])
        if np.linalg.norm(xy - food) < 2.0:
            reached = rows[-1]["t"]
            break
    if video:
        cams[0].save_video(video)
    xy = np.array([[r["x"], r["y"]] for r in rows])
    result = {"seed": seed, "reached": reached is not None, "time_to_food": reached,
              "min_food_dist": float(np.min(np.linalg.norm(xy - food, axis=1))),
              "min_danger_dist": float(np.min(np.linalg.norm(xy - danger_pos, axis=1))),
              "path_mm": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0,
              "mean_valence": float(np.mean([r["valence"] for r in rows])),
              "wall_s": round(time.time() - t0, 1), "rows": rows}
    if record:
        out = pathlib.Path(record); out.mkdir(parents=True, exist_ok=True)
        np.stack(poses).tofile(out / "poses.bin")
        np.concatenate(sp_idx).tofile(out / "spikes_idx.bin")
        np.concatenate(sp_cnt).tofile(out / "spikes_cnt.bin")
        params = {"seed": seed, "seconds": seconds, "reward": dopamine_reward, "punish": dopamine_punish,
                  "octopamine": octopamine, "danger": danger, "heading": float(heading),
                  "food": [float(v) for v in food], "danger_pos": [float(v) for v in danger_pos]}
        (out / "run.json").write_text(json.dumps({"params": params, "frames": len(poses), "frame_dt": 0.01, "windows": windows,
                                                  "rows": rows, "summary": {k: v for k, v in result.items() if k != "rows"}}))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--dopamine-reward", type=float, default=0.0, help="-1 (block) .. 1 (drive PAM)")
    p.add_argument("--dopamine-punish", type=float, default=0.0, help="-1 (block) .. 1 (drive PPL1)")
    p.add_argument("--octopamine", type=float, default=0.0, help="-1 (block) .. 1 (drive OA-VPM4)")
    p.add_argument("--danger", type=float, default=0.0)
    p.add_argument("--food", type=float, nargs=2, help="food x y (mm)")
    p.add_argument("--danger-pos", type=float, nargs=2, help="danger x y (mm)")
    p.add_argument("--heading", type=float, help="start heading (rad); default random from seed")
    p.add_argument("--video")
    p.add_argument("--record", help="directory for sandbox playback files")
    p.add_argument("--out", help="trajectory CSV")
    a = p.parse_args()
    r = trial(a.seed, a.seconds, a.dopamine_reward, a.dopamine_punish, a.octopamine, a.danger, heading=a.heading,
              video=a.video, food=a.food, danger_pos=a.danger_pos, record=a.record,
              log=lambda row: print(f"t={row['t']:.2f}s pos=({row['x']:.1f},{row['y']:.1f}) valence={row['valence']:+.2f}",
                                    flush=True) if int(row["t"] * 40) % 4 == 0 else None)
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=r["rows"][0].keys()); wr.writeheader(); wr.writerows(r["rows"])
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}))
