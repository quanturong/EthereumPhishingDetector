"""
Stage 9 - Upgrade ego-graph tu K=1 len K=2
============================================
Attacks BPH-*/MWR co cau truc 2-3 hop; K=1 lam mat perturbation nodes ngoai hop-1.

Pipeline:
  9a. Extract unique hop-1 neighbors tu hop-1 txs (offline)
  9b. Crawl hop-1 txs cho tung neighbor (cap 500 txs, ~9h)
  9c. Rebuild K=2 ego-graph + apply temporal cutoff

Chay:
  python stage9_k2_upgrade.py --api_key KEY --step 9a
  python stage9_k2_upgrade.py --api_key KEY --step 9b
  python stage9_k2_upgrade.py                 --step 9c
"""

import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from crawl_normal_candidates import etherscan_get, DELAY

BASE_DIR   = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE3_DIR = BASE_DIR / "ethphishgraph2026_stage3_dataset"
STAGE9_DIR = BASE_DIR / "ethphishgraph2026_stage9_k2"
HOP2_TXS_DIR = STAGE9_DIR / "hop2_neighbor_txs"
PHISHING_HOP1_DIR = BASE_DIR / "ethphishgraph2026_stage2_normal_crawl" / "hop1_txs"
NORMAL_HOP1_DIR   = STAGE3_DIR / "normal_hop1_txs"

SPLIT_CSV    = STAGE3_DIR / "temporal_split.csv"
SPLIT_META   = STAGE3_DIR / "temporal_split_meta.json"
UNIQUE_NEIGH = STAGE9_DIR / "unique_hop1_neighbors.csv"
K2_DATASET   = STAGE9_DIR / "ego_graph_k2_dataset.pkl"
MAX_TX_PER_NEIGHBOR = 500


def _load_hop1(addr, label):
    d = PHISHING_HOP1_DIR if label == 1 else NORMAL_HOP1_DIR
    f = d / f"{addr}.json"
    if not f.exists() or f.stat().st_size <= 2: return []
    return json.loads(f.read_text(encoding="utf-8"))


def step_9a_extract_unique():
    STAGE9_DIR.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(SPLIT_CSV)
    print(f"[9a] {len(split)} targets")
    all_neighbors = set()
    per_target = {}
    for _, row in tqdm(split.iterrows(), total=len(split), unit="tgt"):
        addr, label = row["address"].lower(), int(row["label"])
        txs = _load_hop1(addr, label)
        neighs = set()
        for t in txs:
            s = (t.get("from") or "").lower()
            r = (t.get("to")   or "").lower()
            if s and s != addr: neighs.add(s)
            if r and r != addr: neighs.add(r)
        per_target[addr] = neighs
        all_neighbors.update(neighs)
    target_set = set(split["address"].str.lower())
    unique = all_neighbors - target_set
    print(f"[9a] Unique neighbors (excl targets): {len(unique):,}")
    pd.DataFrame({"address": sorted(unique)}).to_csv(UNIQUE_NEIGH, index=False)
    with open(STAGE9_DIR / "per_target_neighbors.pkl", "wb") as f:
        pickle.dump(per_target, f)
    print(f"[9a] Estimated crawl: {len(unique) * 0.25 / 3600:.1f}h")


def step_9b_crawl_neighbors(api_key):
    HOP2_TXS_DIR.mkdir(parents=True, exist_ok=True)
    neighbors = pd.read_csv(UNIQUE_NEIGH)["address"].str.lower().tolist()
    print(f"[9b] Crawl {len(neighbors):,} neighbors (cap {MAX_TX_PER_NEIGHBOR})")
    skipped = 0
    for addr in tqdm(neighbors, unit="addr"):
        out = HOP2_TXS_DIR / f"{addr}.json"
        if out.exists() and out.stat().st_size > 2:
            skipped += 1; continue
        data = etherscan_get({
            "module": "account", "action": "txlist",
            "address": addr, "startblock": 0, "endblock": 99999999,
            "page": 1, "offset": MAX_TX_PER_NEIGHBOR, "sort": "asc",
        }, api_key)
        txs = data.get("result", []) if (data and data.get("status") == "1") else []
        out.write_text(json.dumps(txs), encoding="utf-8")
        time.sleep(DELAY)
    print(f"[9b] Skipped {skipped:,}")


def _build_k2_ego(target, label, cutoff_ts, hop1_neighs, target_txs, neigh_txs_map):
    hop2 = set()
    for neigh, ntxs in neigh_txs_map.items():
        for t in ntxs:
            if int(t.get("timeStamp", 0)) > cutoff_ts: continue
            s = (t.get("from") or "").lower()
            r = (t.get("to")   or "").lower()
            other = r if s == neigh else s
            if other and other != target and other not in hop1_neighs:
                hop2.add(other)
    nodes = {target} | hop1_neighs | hop2
    nodes.discard("")
    if len(nodes) < 2: return None
    a2i = {a: i for i, a in enumerate(sorted(nodes))}
    tgt_idx = a2i[target]

    all_txs, seen = [], set()
    for t in target_txs:
        h = t.get("hash", "")
        if h and h not in seen: seen.add(h); all_txs.append(t)
    for _, ntxs in neigh_txs_map.items():
        for t in ntxs:
            h = t.get("hash", "")
            if h and h not in seen: seen.add(h); all_txs.append(t)

    e_src, e_dst, e_attr = [], [], []
    per_node = {a: {"in": [], "out": []} for a in nodes}
    for t in all_txs:
        ts = int(t.get("timeStamp", 0) or 0)
        if ts > cutoff_ts: continue
        s = (t.get("from") or "").lower()
        r = (t.get("to")   or "").lower()
        if s not in a2i or r not in a2i: continue
        e_src.append(a2i[s]); e_dst.append(a2i[r])
        val = int(t.get("value", "0") or "0")
        gp  = int(t.get("gasPrice", "0") or "0")
        gu  = int(t.get("gasUsed",  "0") or "0")
        e_attr.append([val, ts, gp, gu])
        per_node[s]["out"].append((r, val, ts))
        per_node[r]["in"].append((s, val, ts))
    if not e_src: return None

    n = len(nodes)
    feats = np.zeros((n, 11), dtype=np.float64)
    for a, idx in a2i.items():
        ins, outs = per_node[a]["in"], per_node[a]["out"]
        iv, ov = [v for _, v, _ in ins], [v for _, v, _ in outs]
        ats = [ts for _, _, ts in ins + outs]
        feats[idx] = [
            len(ins), len(outs),
            float(sum(iv)), float(sum(ov)),
            float(np.mean(iv)) if iv else 0.0,
            float(np.mean(ov)) if ov else 0.0,
            float(min(ats)) if ats else 0.0,
            float(max(ats)) if ats else 0.0,
            float(max(ats) - min(ats)) if ats else 0.0,
            float(len({s for s,_,_ in ins})),
            float(len({r for r,_,_ in outs})),
        ]
    return {
        "target": target, "target_idx": tgt_idx, "label": label,
        "cutoff_ts": cutoff_ts,
        "addresses": [a for a, _ in sorted(a2i.items(), key=lambda x: x[1])],
        "edge_index": np.array([e_src, e_dst], dtype=np.int64),
        "edge_attr": np.array(e_attr, dtype=np.float64),
        "node_features": feats,
        "n_nodes": n, "n_edges": len(e_src),
        "n_hop1": len(hop1_neighs & nodes), "n_hop2": len(hop2 & nodes),
    }


_BAD_FILES = []  # track bad files for post-run report


def _load_neighbor_txs(addr):
    f = HOP2_TXS_DIR / f"{addr}.json"
    if not f.exists() or f.stat().st_size <= 2:
        return []
    try:
        with open(f, "r", encoding="utf-8") as fp:
            return json.loads(fp.read())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        _BAD_FILES.append((addr, type(e).__name__))
        return []


def step_9c_build_k2():
    split = pd.read_csv(SPLIT_CSV)
    meta  = json.loads(SPLIT_META.read_text(encoding="utf-8"))
    T1, T2 = meta["T1_ts"], meta["T2_ts"]
    with open(STAGE9_DIR / "per_target_neighbors.pkl", "rb") as f:
        per_target = pickle.load(f)
    T_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600
    cutoffs = {"train": T1, "val": T2, "test": T_end}

    dataset = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in tqdm(split.iterrows(), total=len(split), unit="tgt"):
        addr = row["address"].lower(); label = int(row["label"])
        part = row["partition"]; cutoff = cutoffs[part]
        h1 = per_target.get(addr, set())
        tgt_txs = _load_hop1(addr, label)
        neigh_map = {}
        for n in h1:
            nt = _load_neighbor_txs(n)
            if nt: neigh_map[n] = nt
        g = _build_k2_ego(addr, label, cutoff, h1, tgt_txs, neigh_map)
        if g is None: n_dropped += 1; continue
        g["partition"] = part
        dataset[part].append(g)

    print(f"[9c] Dropped {n_dropped}")
    for p, arr in dataset.items():
        n_ph = sum(g["label"] == 1 for g in arr)
        n_no = sum(g["label"] == 0 for g in arr)
        avg_n = np.mean([g["n_nodes"] for g in arr]) if arr else 0
        avg_e = np.mean([g["n_edges"] for g in arr]) if arr else 0
        avg_h1 = np.mean([g["n_hop1"] for g in arr]) if arr else 0
        avg_h2 = np.mean([g["n_hop2"] for g in arr]) if arr else 0
        print(f"  {p:5s}: {len(arr)} (ph={n_ph}, no={n_no}) avg_nodes={avg_n:.1f} "
              f"(h1={avg_h1:.1f}, h2={avg_h2:.1f}) edges={avg_e:.1f}")

    with open(K2_DATASET, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[9c] Saved -> {K2_DATASET}")

    if _BAD_FILES:
        bad_log = STAGE9_DIR / "bad_files.txt"
        with open(bad_log, "w", encoding="utf-8") as fp:
            for addr, err in _BAD_FILES:
                fp.write(f"{addr}\t{err}\n")
        print(f"[9c] Skipped {len(_BAD_FILES)} bad files -> {bad_log}")
    else:
        print(f"[9c] No bad files.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api_key", default=None)
    p.add_argument("--step", choices=["9a","9b","9c","all"], default="all")
    a = p.parse_args()
    STAGE9_DIR.mkdir(parents=True, exist_ok=True)
    if a.step in ("9a","all"): step_9a_extract_unique()
    if a.step in ("9b","all"):
        if not a.api_key: raise ValueError("Can --api_key cho 9b")
        step_9b_crawl_neighbors(a.api_key)
    if a.step in ("9c","all"): step_9c_build_k2()


if __name__ == "__main__":
    main()