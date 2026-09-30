"""
Stage 3 – Xây dựng dataset EthPhishGraph-2026 (target accounts + ego-graph K=1)
==============================================================================
Pipeline:
  3a. Crawl hop-1 txs cho 3518 normal candidates (Etherscan V2 API)
  3b. Temporal split 60/20/20 dựa trên anchor time (giao dịch đầu tiên)
      -> xác định mốc T1, T2 và gán partition cho từng target
  3c. Dựng ego-graph K=1 với temporal cutoff:
        - Train targets: chỉ dùng tx có timestamp <= T1
        - Val targets:   chỉ dùng tx có timestamp <= T2
        - Test targets:  dùng toàn bộ tx đã thu
      -> lưu edge_index, edge_attr, node_features per target

Cách dùng:
  python stage3_pipeline.py --api_key YOUR_KEY --step 3a   # crawl (~1h, cần mạng)
  python stage3_pipeline.py --step 3b                       # temporal split (offline)
  python stage3_pipeline.py --step 3c                       # ego-graph builder (offline)
  python stage3_pipeline.py --step all --api_key YOUR_KEY

Reuse Etherscan client từ crawl_normal_candidates.py.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse etherscan_get từ script trước
from crawl_normal_candidates import etherscan_get, DELAY, HOP1_DIR as PHISHING_HOP1_DIR

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE1_DIR    = BASE_DIR / "ethphishgraph2026_stage1_outputs"
STAGE2_DIR    = BASE_DIR / "ethphishgraph2026_stage2_normal_crawl"
STAGE3_DIR    = BASE_DIR / "ethphishgraph2026_stage3_dataset"
NORMAL_HOP1_DIR = STAGE3_DIR / "normal_hop1_txs"
EGO_DIR       = STAGE3_DIR / "ego_graphs"

PHISHING_CSV  = STAGE1_DIR / "phishing_final_5_2000.csv"
NORMAL_CSV    = STAGE2_DIR / "normal_candidates.csv"

SPLIT_CSV     = STAGE3_DIR / "temporal_split.csv"
SPLIT_META    = STAGE3_DIR / "temporal_split_meta.json"
DATASET_PKL   = STAGE3_DIR / "ego_graph_dataset.pkl"


# ── 3a. Crawl hop-1 txs cho normal candidates ─────────────────────────────────
def step_3a_crawl_normal_hop1(api_key: str):
    """Cùng logic bước 2a nhưng cho normal candidates. Checkpoint per addr."""
    normal_df = pd.read_csv(NORMAL_CSV)
    addresses = normal_df["address"].str.lower().tolist()
    print(f"[3a] Crawl hop-1 txs cho {len(addresses)} normal candidates...")
    NORMAL_HOP1_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for addr in tqdm(addresses, unit="addr"):
        out_file = NORMAL_HOP1_DIR / f"{addr}.json"
        if out_file.exists() and out_file.stat().st_size > 2:
            skipped += 1
            continue

        txs, page = [], 1
        while True:
            data = etherscan_get({
                "module": "account", "action": "txlist",
                "address": addr, "startblock": 0, "endblock": 99999999,
                "page": page, "offset": 10000, "sort": "asc",
            }, api_key)
            if data is None or data.get("status") != "1":
                break
            batch = data["result"]
            txs.extend(batch)
            if len(batch) < 10000:
                break
            page += 1
            time.sleep(DELAY)

        out_file.write_text(json.dumps(txs), encoding="utf-8")
        time.sleep(DELAY)

    print(f"[3a] Xong. Skipped {skipped} địa chỉ đã crawl.")


# ── Tiện ích: load hop-1 txs của 1 target ────────────────────────────────────
def _load_hop1_txs(addr: str, label: int) -> list[dict]:
    src_dir = PHISHING_HOP1_DIR if label == 1 else NORMAL_HOP1_DIR
    f = src_dir / f"{addr}.json"
    if not f.exists() or f.stat().st_size <= 2:
        return []
    return json.loads(f.read_text(encoding="utf-8"))


# ── 3b. Temporal split ────────────────────────────────────────────────────────
def step_3b_temporal_split():
    """
    - Anchor time = timestamp của giao dịch đầu tiên trong hop-1
    - Sort toàn bộ 7036 target theo anchor_ts, chia 60/20/20
    - T1 = anchor_ts của target ranked 60%
    - T2 = anchor_ts của target ranked 80%
    """
    phishing = pd.read_csv(PHISHING_CSV)[["address"]]
    phishing["label"] = 1
    normal = pd.read_csv(NORMAL_CSV)[["address"]]
    normal["label"] = 0
    targets = pd.concat([phishing, normal], ignore_index=True)
    targets["address"] = targets["address"].str.lower()

    print(f"[3b] Tính anchor_ts cho {len(targets):,} target accounts...")
    anchor_ts, tx_counts = [], []
    for _, row in tqdm(targets.iterrows(), total=len(targets), unit="target"):
        txs = _load_hop1_txs(row["address"], row["label"])
        if not txs:
            anchor_ts.append(None)
            tx_counts.append(0)
            continue
        # timestamps là string trong Etherscan response
        ts = [int(t["timeStamp"]) for t in txs if t.get("timeStamp")]
        anchor_ts.append(min(ts) if ts else None)
        tx_counts.append(len(txs))

    targets["anchor_ts"] = anchor_ts
    targets["hop1_tx_count"] = tx_counts

    # Loại target không có tx (defensive)
    valid = targets.dropna(subset=["anchor_ts"]).copy()
    print(f"[3b] Target hợp lệ (có anchor_ts): {len(valid):,} / {len(targets):,}")
    print(f"     Loại {len(targets) - len(valid):,} target không có hop-1 tx.")

    valid = valid.sort_values("anchor_ts").reset_index(drop=True)
    n = len(valid)
    i_t1 = int(n * 0.6)
    i_t2 = int(n * 0.8)
    T1_ts = int(valid.iloc[i_t1]["anchor_ts"])
    T2_ts = int(valid.iloc[i_t2]["anchor_ts"])

    valid["partition"] = "test"
    valid.loc[valid.index < i_t1, "partition"] = "train"
    valid.loc[(valid.index >= i_t1) & (valid.index < i_t2), "partition"] = "val"

    valid.to_csv(SPLIT_CSV, index=False)

    meta = {
        "n_total":       int(n),
        "n_train":       int((valid["partition"] == "train").sum()),
        "n_val":         int((valid["partition"] == "val").sum()),
        "n_test":        int((valid["partition"] == "test").sum()),
        "T1_ts":         T1_ts,
        "T2_ts":         T2_ts,
        "T1_iso":        pd.to_datetime(T1_ts, unit="s").isoformat(),
        "T2_iso":        pd.to_datetime(T2_ts, unit="s").isoformat(),
        "anchor_min_ts": int(valid["anchor_ts"].min()),
        "anchor_max_ts": int(valid["anchor_ts"].max()),
    }
    SPLIT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[3b] Split done -> {SPLIT_CSV}")
    print(f"     T1 = {meta['T1_iso']}")
    print(f"     T2 = {meta['T2_iso']}")
    print(f"     train={meta['n_train']}, val={meta['n_val']}, test={meta['n_test']}")
    print()
    print("     Class balance mỗi partition:")
    print(valid.groupby(["partition", "label"]).size().unstack(fill_value=0).to_string())


# ── 3c. Ego-graph K=1 với temporal cutoff ─────────────────────────────────────
NODE_FEATURES = [
    "in_count", "out_count",
    "in_value_sum", "out_value_sum",
    "in_value_mean", "out_value_mean",
    "first_ts", "last_ts", "active_span",
    "unique_in_peers", "unique_out_peers",
]
EDGE_FEATURES = ["value", "timestamp", "gas_price", "gas_used"]


def _compute_node_features(txs_for_node: dict) -> dict:
    """txs_for_node = {addr: {'in': [...], 'out': [...]}}. Trả về feature dict."""
    ...  # xử lý dưới đây


def _build_one_ego_graph(target: str, label: int, cutoff_ts: int, raw_txs: list[dict]) -> dict | None:
    """Trả về dict biểu diễn 1 ego-graph, hoặc None nếu rỗng sau cutoff."""
    # Lọc theo cutoff
    txs = [t for t in raw_txs if int(t.get("timeStamp", 0)) <= cutoff_ts]
    if not txs:
        return None

    # Collect nodes
    nodes: set[str] = {target}
    for t in txs:
        s = (t.get("from") or "").lower()
        r = (t.get("to")   or "").lower()
        if s: nodes.add(s)
        if r: nodes.add(r)
    nodes.discard("")

    addr_to_idx = {a: i for i, a in enumerate(sorted(nodes))}
    target_idx = addr_to_idx[target]

    # Edges
    edge_src, edge_dst = [], []
    edge_attr = []
    for t in txs:
        s = (t.get("from") or "").lower()
        r = (t.get("to")   or "").lower()
        if not s or not r or s not in addr_to_idx or r not in addr_to_idx:
            continue
        edge_src.append(addr_to_idx[s])
        edge_dst.append(addr_to_idx[r])
        value_wei = int(t.get("value", "0") or "0")
        ts        = int(t.get("timeStamp", "0") or "0")
        gp        = int(t.get("gasPrice", "0")  or "0")
        gu        = int(t.get("gasUsed",  "0")  or "0")
        edge_attr.append([value_wei, ts, gp, gu])

    if not edge_src:
        return None

    # Node features
    per_node = {a: {"in": [], "out": []} for a in nodes}
    for t in txs:
        s = (t.get("from") or "").lower()
        r = (t.get("to")   or "").lower()
        val = int(t.get("value", "0") or "0")
        ts  = int(t.get("timeStamp", "0") or "0")
        if s in per_node:
            per_node[s]["out"].append((r, val, ts))
        if r in per_node:
            per_node[r]["in"].append((s, val, ts))

    n_nodes = len(nodes)
    node_feats = np.zeros((n_nodes, len(NODE_FEATURES)), dtype=np.float64)
    for a, idx in addr_to_idx.items():
        ins  = per_node[a]["in"]
        outs = per_node[a]["out"]
        in_vals  = [v for _, v, _ in ins]
        out_vals = [v for _, v, _ in outs]
        all_ts   = [ts for _, _, ts in ins + outs]
        node_feats[idx] = [
            len(ins),
            len(outs),
            float(sum(in_vals)),
            float(sum(out_vals)),
            float(np.mean(in_vals))  if in_vals  else 0.0,
            float(np.mean(out_vals)) if out_vals else 0.0,
            float(min(all_ts)) if all_ts else 0.0,
            float(max(all_ts)) if all_ts else 0.0,
            float(max(all_ts) - min(all_ts)) if all_ts else 0.0,
            float(len({s for s, _, _ in ins})),
            float(len({r for r, _, _ in outs})),
        ]

    return {
        "target":       target,
        "target_idx":   target_idx,
        "label":        label,
        "cutoff_ts":    cutoff_ts,
        "addresses":    [a for a, _ in sorted(addr_to_idx.items(), key=lambda x: x[1])],
        "edge_index":   np.array([edge_src, edge_dst], dtype=np.int64),  # [2, E]
        "edge_attr":    np.array(edge_attr, dtype=np.float64),           # [E, 4]
        "node_features": node_feats,                                     # [N, 11]
        "n_nodes":      n_nodes,
        "n_edges":      len(edge_src),
    }


def step_3c_build_ego_graphs():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError("[3c] Cần chạy bước 3b trước.")
    split_df = pd.read_csv(SPLIT_CSV)
    meta     = json.loads(SPLIT_META.read_text(encoding="utf-8"))
    T1, T2   = meta["T1_ts"], meta["T2_ts"]

    # Cutoff cho mỗi partition
    T_end = int(split_df["anchor_ts"].max()) + 10 * 365 * 24 * 3600  # +10 năm = "vô cực"
    part_cutoff = {"train": T1, "val": T2, "test": T_end}

    print(f"[3c] Cutoffs: train<=T1, val<=T2, test<={pd.to_datetime(T_end, unit='s').year}")

    dataset = {"train": [], "val": [], "test": []}
    n_dropped = 0

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), unit="target"):
        addr   = row["address"]
        label  = int(row["label"])
        part   = row["partition"]
        cutoff = part_cutoff[part]

        raw = _load_hop1_txs(addr, label)
        g   = _build_one_ego_graph(addr, label, cutoff, raw)
        if g is None:
            n_dropped += 1
            continue
        g["partition"] = part
        dataset[part].append(g)

    print(f"[3c] Dropped (rỗng sau cutoff): {n_dropped}")
    for p, arr in dataset.items():
        n_phish = sum(g["label"] == 1 for g in arr)
        n_norm  = sum(g["label"] == 0 for g in arr)
        avg_n   = np.mean([g["n_nodes"] for g in arr]) if arr else 0
        avg_e   = np.mean([g["n_edges"] for g in arr]) if arr else 0
        print(f"  {p:5s}: {len(arr):>4} ego-graphs  "
              f"(phish={n_phish}, norm={n_norm})  "
              f"avg_nodes={avg_n:.1f}, avg_edges={avg_e:.1f}")

    with open(DATASET_PKL, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[3c] Saved -> {DATASET_PKL}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3 pipeline - EthPhishGraph-2026")
    parser.add_argument("--api_key", default=None, help="Etherscan API key (bắt buộc cho 3a)")
    parser.add_argument("--step", choices=["3a", "3b", "3c", "all"], default="all")
    args = parser.parse_args()

    STAGE3_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("3a", "all"):
        if not args.api_key:
            raise ValueError("Cần --api_key cho bước 3a")
        step_3a_crawl_normal_hop1(args.api_key)
    if args.step in ("3b", "all"):
        step_3b_temporal_split()
    if args.step in ("3c", "all"):
        step_3c_build_ego_graphs()


if __name__ == "__main__":
    main()
