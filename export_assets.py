"""Export assets for the 3D sandbox (sandbox/assets/):
  fly_verts.bin / fly_faces.bin  NeuroMechFly meshes straight from the MuJoCo model
  walk.bin                       1 s straight-walking clip (12 gait cycles at 12 Hz -> loops), per-geom
                                 pose relative to the fly's yaw frame, 10 ms frames: [x,y,z, 9 rot]
  neurons.bin                    FlyWire neuron positions (um, float32 xyz) + group ids (uint8)
  fly.json                       mesh index, walk metadata, live-motion calibration, group legend
Run: python export_assets.py"""
import json, pathlib
import numpy as np, pandas as pd
import run
from brain import DATA

OUT = pathlib.Path(__file__).parent / "sandbox" / "assets"
FRAME_STEPS = int(0.01 / run.BODY_DT)          # 10 ms per exported frame


def mesh_geoms(m):
    return [g for g in range(m.ngeom) if m.geom_type[g] == 7 and m.geom_dataid[g] >= 0]


def yaw_frame(sim):
    d = sim.physics.named.data
    p = d.xpos[f"{sim.fly.name}/Thorax"].copy()
    fwd = d.xmat[f"{sim.fly.name}/Thorax"].reshape(3, 3)[:, 0]
    yaw = np.arctan2(fwd[1], fwd[0])
    return p, yaw


def geom_poses(sim, geoms, p, yaw):
    """Geom poses in the fly's yaw frame (x,y relative to thorax; z absolute)."""
    d = sim.physics.data
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])        # world -> yaw frame
    out = np.zeros((len(geoms), 12), np.float32)
    for k, g in enumerate(geoms):
        rel = d.geom_xpos[g] - np.array([p[0], p[1], 0])
        out[k, :3] = R @ rel
        out[k, 3:] = (R @ d.geom_xmat[g].reshape(3, 3)).ravel()
    return out


def export_meshes(sim):
    m = sim.physics.model
    geoms = mesh_geoms(m)
    verts, faces, index = [], [], []
    vo = fo = 0
    for g in geoms:
        mid = m.geom_dataid[g]
        va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts.append(m.mesh_vert[va:va + vn].astype(np.float32))
        faces.append(m.mesh_face[fa:fa + fn].astype(np.uint32))
        index.append({"name": m.id2name(g, "geom").split("/")[-1], "v": [int(vo), int(vn)], "f": [int(fo), int(fn)]})
        vo += vn; fo += fn
    np.concatenate(verts).tofile(OUT / "fly_verts.bin")
    np.concatenate(faces).tofile(OUT / "fly_faces.bin")
    return geoms, index


def record_walk(geoms):
    sim, _ = run.make_sim(0.0, None)
    sim.reset(seed=0)
    for _ in range(int(0.5 / run.BODY_DT)):             # settle into the gait
        sim.step(np.array([1.0, 1.0]))
    frames = []
    for i in range(int(1.0 / run.BODY_DT)):
        if i % FRAME_STEPS == 0:
            p, yaw = yaw_frame(sim)
            frames.append(geom_poses(sim, geoms, p, yaw))
        sim.step(np.array([1.0, 1.0]))
    np.stack(frames).tofile(OUT / "walk.bin")
    return len(frames)


def calibrate_motion():
    """Fit live kinematics: speed = a * mean(drive), yaw rate = b * (dR - dL), from the physics body."""
    rows = []
    for act in ([1.0, 1.0], [1.2, 1.2], [1.0, 0.6], [0.6, 1.0], [1.0, 0.2], [0.2, 1.0]):
        sim, _ = run.make_sim(0.0, None)
        obs, _ = sim.reset(seed=0)
        for _ in range(int(0.3 / run.BODY_DT)):
            obs, *_ = sim.step(np.array(act))
        p0, h0 = obs["fly"][0, :2].copy(), obs["cardinal_vectors"][0][:2].copy()
        for _ in range(int(1.0 / run.BODY_DT)):
            obs, *_ = sim.step(np.array(act))
        p1, h1 = obs["fly"][0, :2], obs["cardinal_vectors"][0][:2]
        dyaw = np.arctan2(h0[0] * h1[1] - h0[1] * h1[0], h0 @ h1)
        rows.append((np.mean(act), act[1] - act[0], np.linalg.norm(p1 - p0), dyaw))
        print(f"  action {act}: {rows[-1][2]:.1f} mm/s, {np.degrees(dyaw):+.0f} deg/s")
    r = np.array(rows)
    a = float(np.sum(r[:, 0] * r[:, 2]) / np.sum(r[:, 0] ** 2))
    b = float(np.sum(r[:, 1] * r[:, 3]) / np.sum(r[:, 1] ** 2))
    return {"speed_per_drive": a, "yaw_per_diff": b}


def export_neurons():
    ids = pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index.astype(str)
    ann = (pd.read_csv(DATA / "annotations_783.tsv", sep="\t", dtype=str, low_memory=False)
           .drop_duplicates("root_id").set_index("root_id").reindex(ids))
    xyz = ann[["pos_x", "pos_y", "pos_z"]].astype(float).to_numpy() * np.array([4, 4, 40]) / 1000.0   # voxel -> um
    xyz = np.nan_to_num(xyz, nan=np.nanmean(xyz, axis=0)).astype(np.float32)
    g = json.loads((DATA / "groups.json").read_text())["groups"]
    legend = [("other", "#3a4458"), ("food ORN", "#ff9a3c"), ("danger ORN", "#3ca0ff"), ("Kenyon cell", "#9b6bff"),
              ("reward DA (PAM)", "#2ee88a"), ("punish DA (PPL1)", "#ff4b5c"), ("approach MBON", "#c6ff3c"),
              ("avoid MBON", "#ff3cd2"), ("other MBON", "#e0b0ff"), ("descending", "#ffffff"), ("octopamine", "#ffe03c")]
    gid = np.zeros(len(ids), np.uint8)
    put = lambda names, k: [gid.__setitem__(np.asarray(g[n]), k) for n in names if n in g]
    put(["KC"], 3)
    put([n for n in g if n.startswith("MBON")], 8)
    put(run.APPROACH, 6); put(run.AVOID, 7)
    put([n for n in g if n.startswith("PAM")], 4)
    put([n for n in g if n.startswith("PPL1")], 5)
    desc = (ann.super_class.fillna("") == "descending").to_numpy()
    gid[desc] = 9
    put(["food_L", "food_R"], 1); put(["danger_L", "danger_R"], 2); put(["OA_VPM4"], 10)
    xyz.tofile(OUT / "neurons.bin")
    gid.tofile(OUT / "neuron_groups.bin")
    return [{"name": n, "color": c, "count": int((gid == k).sum())} for k, (n, c) in enumerate(legend)]


def decision_meta():
    """Neurons behind the decision readout + the data-derived dopamine compartments (for the 'why' view)."""
    import brain
    g = json.loads((DATA / "groups.json").read_text())
    return {"mbon_types": {m: g["groups"][m] for m in run.APPROACH + run.AVOID}, "approach": run.APPROACH, "avoid": run.AVOID,
            "compartments": g["compartments"], "D50": brain.D50}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim, _ = run.make_sim(0.0, None)
    obs, _ = sim.reset(seed=0)
    geoms, index = export_meshes(sim)
    p, yaw = yaw_frame(sim)
    nd = sim.physics.named.data
    antenna = {s: (nd.xpos[f"{sim.fly.name}/{s}Funiculus"] - p)[:2].tolist() for s in "LR"}
    print("meshes", len(index), "antenna offsets (mm)", antenna)
    frames = record_walk(geoms)
    print("walk frames", frames)
    motion = calibrate_motion()
    print("motion", motion)
    legend = export_neurons()
    meta = {"meshes": index, "walk": {"frames": frames, "dt": 0.01, "geoms": len(geoms)},
            "antenna": antenna, "motion": motion, "legend": legend, "decision": decision_meta(),
            "reflex": {"G_FOOD": run.G_FOOD, "G_DANGER": run.G_DANGER, "DRIVE": run.DRIVE, "AROUSAL": run.AROUSAL,
                       "KNOB_HZ": run.KNOB_HZ, "ORN_MAX_HZ": run.ORN_MAX_HZ, "ORN_HALF": run.ORN_HALF}}
    (OUT / "fly.json").write_text(json.dumps(meta))
    print("wrote", OUT)


if __name__ == "__main__":
    import sys
    if "--neurons-only" in sys.argv:          # quick re-export of brain-view positions/groups
        meta_path = OUT / "fly.json"
        meta = json.loads(meta_path.read_text()); meta["legend"] = export_neurons(); meta["decision"] = decision_meta()
        meta_path.write_text(json.dumps(meta))
    else:
        main()
