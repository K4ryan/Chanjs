"""Download FlyWire v783 (Shiu et al. LIF weights + FlyWire annotations) and build
data/circuit.npz (CSR graph + flags) and data/groups.json (named neuron groups and
the data-derived dopamine compartment table). Run: python fetch_data.py"""
import hashlib, json, pathlib, urllib.request
import numpy as np, pandas as pd

DATA = pathlib.Path(__file__).parent / "data"
SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/"
ANN = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/"
SOURCES = {   # pinned commits: the exact bytes every result here was produced with
    "Completeness_783.csv": SHIU + "Completeness_783.csv",
    "Connectivity_783.parquet": SHIU + "Connectivity_783.parquet",
    "annotations_783.tsv": ANN + "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}
SHA256 = {
    "Completeness_783.csv": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    "Connectivity_783.parquet": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    "annotations_783.tsv": "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
}
COMPARTMENT_MIN = 0.2  # a DAN type "innervates" an MBON type if >=20% of its DAN->MBON synapses go there


def fetch():
    DATA.mkdir(exist_ok=True)
    hashes = {}
    for name, url in SOURCES.items():
        path = DATA / name
        if not path.exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, path)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != SHA256[name]:
            raise SystemExit(f"{name}: sha256 {h} != expected {SHA256[name]}; delete data/{name} and rerun")
        hashes[name] = {"url": url, "sha256": h}
    return hashes


def main():
    hashes = fetch()
    ids = pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index.astype(str)
    n = len(ids)
    ann = (pd.read_csv(DATA / "annotations_783.tsv", sep="\t", dtype=str, low_memory=False)
           .drop_duplicates("root_id").set_index("root_id").reindex(ids).fillna(""))
    ct, ht, cls, side = (ann[c].to_numpy(dtype=str) for c in ("cell_type", "hemibrain_type", "cell_class", "side"))
    e = pd.read_parquet(DATA / "Connectivity_783.parquet").sort_values(["Presynaptic_Index", "Postsynaptic_Index"])
    pre, post = e.Presynaptic_Index.values, e.Postsynaptic_Index.values
    w = e["Excitatory x Connectivity"].values.astype(np.float32)
    assert (e.Presynaptic_ID.astype(str).values == ids[pre]).all()

    # Sensory neurotransmitter predictions are unreliable (ORNs often "serotonin"/"gaba"): force excitatory.
    sensory = ann.super_class.values == "sensory"
    w[sensory[pre]] = np.abs(w[sensory[pre]])

    def pick(mask, name):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise SystemExit(f"empty group: {name}")
        return idx

    def typed(t):  # FlyWire cell_type is authoritative; hemibrain_type as fallback
        return (ct == t) | ((ct == "") & (ht == t))

    groups = {}
    for s, S in (("left", "L"), ("right", "R")):
        on = side == s
        groups[f"food_{S}"] = pick(np.isin(ht, ["ORN_DM1", "ORN_DM4", "ORN_VA2"]) & on, f"food_{S}")
        groups[f"danger_{S}"] = pick(np.isin(ht, ["ORN_DA2", "ORN_V"]) & on, f"danger_{S}")
        for dn in ("DNa02", "DNa01", "DNp09", "MDN"):
            groups[f"{dn}_{S}"] = pick(typed(dn) & on, f"{dn}_{S}")
    groups["OA_VPM4"] = pick(typed("OA-VPM4"), "OA-VPM4")
    groups["KC"] = pick(cls == "Kenyon_Cell", "KC")
    dan_type = np.where(ct != "", ct, ht)
    mbon_type = ht
    dans = sorted(set(dan_type[cls == "DAN"]))
    for d in dans:
        groups[d] = pick((cls == "DAN") & (dan_type == d), d)
    mbons = sorted(set(mbon_type[cls == "MBON"]))
    for m in mbons:
        groups[m] = pick((cls == "MBON") & (mbon_type == m), m)

    # Data-derived compartments: DAN type -> MBON types it synapses onto (fraction of its MBON output).
    dm = (cls[pre] == "DAN") & (cls[post] == "MBON")
    t = pd.DataFrame({"dan": dan_type[pre[dm]], "mbon": mbon_type[post[dm]], "n": e.Connectivity.values[dm]})
    t = t.groupby(["dan", "mbon"]).n.sum().reset_index()
    t["frac"] = t.n / t.groupby("dan").n.transform("sum")
    comp = t[t.frac >= COMPARTMENT_MIN].sort_values(["dan", "frac"], ascending=[True, False])
    print("Dopamine compartments (DAN type -> MBON type, share of its MBON synapses):")
    for d, g in comp.groupby("dan"):
        print(f"  {d:7s} -> " + ", ".join(f"{r.mbon} {r.frac:.0%}" for r in g.itertuples()))

    # Neuromodulator neurons: their fast synapses are removed; they act only through the slow channel.
    modulatory = (cls == "DAN") | np.char.startswith(ct, "OA-") | np.char.startswith(ht, "OA-")
    kc_mbon = (cls[pre] == "Kenyon_Cell") & (cls[post] == "MBON")

    offsets = np.zeros(n + 1, np.int64)
    np.add.at(offsets, pre + 1, 1)
    offsets = np.cumsum(offsets)
    np.savez_compressed(
        DATA / "circuit.npz", offsets=offsets, targets=post.astype(np.int32), weights=w,
        modulatory_pre=modulatory[pre], kc_mbon_edge=np.where(kc_mbon)[0], kc_mbon_type=mbon_type[post[kc_mbon]],
        ids=ids.values.astype(np.int64))
    out = {"groups": {k: v.tolist() for k, v in groups.items()},
           "compartments": comp[["dan", "mbon", "frac"]].to_dict("records"),
           "pam": [d for d in dans if d.startswith("PAM")], "ppl1": [d for d in dans if d.startswith("PPL1")],
           "sources": hashes, "neurons": n, "edges": int(len(post))}
    (DATA / "groups.json").write_text(json.dumps(out))
    print(f"{n} neurons, {len(post)} edges, {modulatory.sum()} modulatory neurons silenced (fast), "
          f"{kc_mbon.sum()} KC->MBON edges")
    for k in [k for k in groups if not k.startswith(("PAM", "PPL", "MBON"))]:
        print(f"  {k:10s} {len(groups[k])}")


if __name__ == "__main__":
    main()
