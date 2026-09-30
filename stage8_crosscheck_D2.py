"""
Stage 8 - Cross-check tren dataset D2 (XBLOCK Phishing Second-order)
=====================================================================
Theo dong 57, 61, 75 de cuong:
  EthereumD2 dong vai tro tap doi chieu de tai lap PEAE-GNN va kiem chung
  ket qua tren EthPhishGraph-2026.

Nguon:  http://xblock.pro/ethereum/
Format:
  dataset/D2/
    钓鱼一阶节点/     (phishing hop-1)     1660 CSV files
    非钓鱼一阶节点/   (normal hop-1)       1700 CSV files
    钓鱼二阶节点/     (phishing hop-2)     folders per target
    非钓鱼二阶节点/   (normal hop-2)       folders per target

CSV columns: TxHash, BlockHeight, TimeStamp, From, To, Value, ContractAddress, Input, isError
Value: ETH (float), khac EthPhishGraph-2026 dung wei.

Pipeline:
  8a. Load 1660 phishing + 1700 normal target -> chi dung hop-1 (K=1 ego-graph)
  8b. Chuan hoa Value ETH -> wei (x 1e18)
  8c. Filter tx count [5, 2000] tuong tu quy tac EthPhishGraph-2026
  8d. Account-level split 60/20/20 theo temporal ordering
  8e. Build ego-graph K=1, tinh 11 node features + 4 edge features
  8f. Train 3 baseline models (GraphSAGE, GATv2, PEAE-GNN) x 3 seeds
  8g. Report clean metrics de so sanh voi EthPhishGraph-2026 (Bang 4.6)
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stage5_train_baselines import (
    MODELS, compute_feature_stats, to_pyg, train_one, evaluate,
)

# --- Paths -------------------------------------------------------------------
BASE_DIR    = Path(r"E:\EthereumPhishingDetection\dataset")
D2_DIR      = BASE_DIR / "D2"
PHISH_HOP1  = D2_DIR / "钓鱼一阶节点"       # 1660 CSVs
NORM_HOP1   = D2_DIR / "非钓鱼一阶节点"     # 1700 CSVs

STAGE8_DIR  = BASE_DIR / "EthPhishGraph-2026" / "ethphishgraph2026_stage8_crosscheck_D2"
D2_PKL      = STAGE8_DIR / "d2_ego_graphs.pkl"
D2_META     = STAGE8_DIR / "d2_meta.json"


# --- D2 loading --------------------------------------------------------------
def _csv_to_txs(csv_path: Path) -> list[dict]:
    """
    Doc 1 CSV cua D2, convert thanh danh sach tx format tuong thich
    voi _build_one_ego_graph (giong Etherscan V2 format).
    """
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
    except Exception:
        return []
    if df.empty:
        return []
    txs = []
    for _, row in df.iterrows():
        try:
            value_eth = float(row.get("Value", 0) or 0)
            value_wei = int(value_eth * 1e18)
            ts = int(row.get("TimeStamp", 0) or 0)
            if ts == 0:
                continue
            txs.append({
                "from":      str(row.get("From", "") or "").lower(),
                "to":        str(row.get("To",   "") or "").lower(),
                "value":     str(value_wei),
                "timeStamp": str(ts),
                "gasPrice":  "20000000000",  # placeholder ~20 gwei
                "gasUsed":   "21000",         # placeholder standard transfer
                "isError":   str(row.get("isError", 0)),
                "hash":      str(row.get("TxHash", "")),
            })
        except (ValueError, TypeError):
            continue
    return txs


def _build_ego_graph_from_txs(target: str, label: int, txs: list[dict],
                              cutoff_ts: int) -> dict | None:
    """
    Reimplementation cua stage3._build_one_ego_graph nhung cho D2 (khong import
    de tranh phu thuoc paths). Cung 11 node features + 4 edge features.
    """
    if not txs:
        return None
    filtered = [t for t in txs if int(t.get("timeStamp", 0)) <= cutoff_ts]
    if not filtered:
        return None

    nodes = {target}
    for t in filtered:
        if t.get("from"): nodes.add(t["from"])
        if t.get("to"):   nodes.add(t["to"])
    nodes.discard("")
    addr_to_idx = {a: i for i, a in enumerate(sorted(nodes))}
    target_idx = addr_to_idx[target]

    edge_src, edge_dst, edge_attr = [], [], []
    for t in filtered:
        s = t.get("from", "").lower()
        r = t.get("to",   "").lower()
        if not s or not r or s not in addr_to_idx or r not in addr_to_idx:
            continue
        edge_src.append(addr_to_idx[s])
        edge_dst.append(addr_to_idx[r])
        edge_attr.append([
            int(t.get("value", "0") or "0"),
            int(t.get("timeStamp", "0") or "0"),
            int(t.get("gasPrice", "0") or "0"),
            int(t.get("gasUsed",  "0") or "0"),
        ])
    if not edge_src:
        return None

    per_node = {a: {"in": [], "out": []} for a in nodes}
    for t in filtered:
        s = t.get("from", "").lower()
        r = t.get("to",   "").lower()
        v = int(t.get("value", "0") or "0")
        ts = int(t.get("timeStamp", "0") or "0")
        if s in per_node:
            per_node[s]["out"].append((r, v, ts))
        if r in per_node:
            per_node[r]["in"].append((s, v, ts))

    n = len(nodes)
    feats = np.zeros((n, 11), dtype=np.float64)
    for a, idx in addr_to_idx.items():
        ins = per_node[a]["in"]
        outs = per_node[a]["out"]
        in_vals  = [v for _, v, _ in ins]
        out_vals = [v for _, v, _ in outs]
        all_ts   = [ts for _, _, ts in ins + outs]
        feats[idx] = [
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
        "target":         target,
        "target_idx":     target_idx,
        "label":          label,
        "cutoff_ts":      cutoff_ts,
        "addresses":      [a for a, _ in sorted(addr_to_idx.items(), key=lambda x: x[1])],
        "edge_index":     np.array([edge_src, edge_dst], dtype=np.int64),
        "edge_attr":      np.array(edge_attr, dtype=np.float64),
        "node_features":  feats,
        "n_nodes":        n,
        "n_edges":        len(edge_src),
    }


# --- Step 8a-e: build D2 dataset ---------------------------------------------
def build_d2_dataset(min_tx=5, max_tx=2000) -> dict:
    STAGE8_DIR.mkdir(parents=True, exist_ok=True)

    all_graphs = []
    stats_summary = {
        "raw_phishing":  0, "raw_normal":  0,
        "kept_phishing": 0, "kept_normal": 0,
        "dropped_low":   0, "dropped_high": 0, "dropped_empty": 0,
    }

    # T_end tam thoi (D2 crawl toi 2019/12/31 - overshoot 5 nam de an toan)
    T_end = int(pd.Timestamp("2030-01-01").timestamp())

    for label, folder, key_kept in [
        (1, PHISH_HOP1, "kept_phishing"),
        (0, NORM_HOP1,  "kept_normal"),
    ]:
        files = list(folder.glob("*.csv"))
        raw_key = "raw_phishing" if label == 1 else "raw_normal"
        stats_summary[raw_key] = len(files)
        print(f"[8a] Loading {len(files)} {'phishing' if label else 'normal'} CSVs...")
        for f in tqdm(files, unit="file"):
            target = f.stem.lower()
            txs = _csv_to_txs(f)
            if not txs:
                stats_summary["dropped_empty"] += 1
                continue
            n_tx = len(txs)
            if n_tx < min_tx:
                stats_summary["dropped_low"] += 1
                continue
            if n_tx > max_tx:
                stats_summary["dropped_high"] += 1
                continue
            g = _build_ego_graph_from_txs(target, label, txs, T_end)
            if g is None:
                stats_summary["dropped_empty"] += 1
                continue
            # Anchor time = timestamp giao dich dau tien
            g["anchor_ts"] = int(min(int(t["timeStamp"]) for t in txs))
            g["hop1_tx_count"] = n_tx
            all_graphs.append(g)
            stats_summary[key_kept] += 1

    print(f"\n[8a] Stats:")
    for k, v in stats_summary.items():
        print(f"  {k:20s}: {v}")

    # 8d. Temporal split 60/20/20 theo anchor_ts
    all_graphs.sort(key=lambda g: g["anchor_ts"])
    n = len(all_graphs)
    i_t1 = int(n * 0.6)
    i_t2 = int(n * 0.8)
    T1 = int(all_graphs[i_t1]["anchor_ts"])
    T2 = int(all_graphs[i_t2]["anchor_ts"])

    for i, g in enumerate(all_graphs):
        if   i < i_t1: g["partition"] = "train"
        elif i < i_t2: g["partition"] = "val"
        else:          g["partition"] = "test"

    dataset = {
        "train": [g for g in all_graphs if g["partition"] == "train"],
        "val":   [g for g in all_graphs if g["partition"] == "val"],
        "test":  [g for g in all_graphs if g["partition"] == "test"],
    }
    for part, arr in dataset.items():
        n_ph = sum(g["label"] == 1 for g in arr)
        n_no = sum(g["label"] == 0 for g in arr)
        avg_n = np.mean([g["n_nodes"] for g in arr]) if arr else 0
        avg_e = np.mean([g["n_edges"] for g in arr]) if arr else 0
        print(f"  {part:5s}: {len(arr):>4} (phish={n_ph}, norm={n_no})  "
              f"avg_nodes={avg_n:.1f}, avg_edges={avg_e:.1f}")

    meta = {
        "n_total": n, "n_train": len(dataset["train"]),
        "n_val": len(dataset["val"]), "n_test": len(dataset["test"]),
        "T1_ts": T1, "T2_ts": T2,
        "T1_iso": pd.to_datetime(T1, unit="s").isoformat(),
        "T2_iso": pd.to_datetime(T2, unit="s").isoformat(),
        "stats": stats_summary,
    }
    print(f"\n  T1 = {meta['T1_iso']}")
    print(f"  T2 = {meta['T2_iso']}")

    with open(D2_PKL, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(D2_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[8a-e] Saved -> {D2_PKL}")
    return dataset


# --- Step 8f-g: train baselines and compare ----------------------------------
def train_baselines_on_d2(epochs=100, seeds=(42, 43, 44), bs=64, patience=15):
    with open(D2_PKL, "rb") as f:
        raw = pickle.load(f)
    print(f"Loaded D2: train={len(raw['train'])}, val={len(raw['val'])}, test={len(raw['test'])}")

    stats = compute_feature_stats(raw["train"])
    train_ds = [to_pyg(g, stats) for g in raw["train"]]
    val_ds   = [to_pyg(g, stats) for g in raw["val"]]
    test_ds  = [to_pyg(g, stats) for g in raw["test"]]
    in_dim = train_ds[0].x.size(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}
    for m in ["graphsage", "gatv2", "peaegnn"]:
        print(f"\n{'='*70}\n Model: {m.upper()} on D2\n{'='*70}")
        seed_results = []
        for s in seeds:
            print(f"\n--- seed={s} ---")
            r = train_one(m, train_ds, val_ds, test_ds, in_dim,
                          epochs=epochs, bs=bs, patience=patience,
                          device=device, seed=s)
            seed_results.append(r)
        agg = {}
        for k in seed_results[0]["test"]:
            vals = [r["test"][k] for r in seed_results]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        all_results[m] = {"agg": agg}
        print(f"\n>>> {m.upper()} on D2 (n_seeds={len(seeds)}):")
        for k, v in agg.items():
            print(f"    {k:15s} {v['mean']:.4f} ± {v['std']:.4f}")

    out = STAGE8_DIR / "d2_baseline_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out}")

    # Summary comparison table
    print(f"\n{'='*80}")
    print(" CROSS-CHECK: EthPhishGraph-2026 vs D2 (mean ± std)")
    print(f"{'='*80}")
    print(f"{'Model':<12s} {'Dataset':<20s} {'MacroF1':<20s} {'PR-AUC':<20s} {'PhishRecall':<20s}")

    # EthPhishGraph-2026 results (từ Bảng 4.6, hard-coded để so sánh trực tiếp)
    ephg_results = {
        "graphsage": {"macro_f1": (0.7831, 0.0120), "pr_auc": (0.8776, 0.0061), "rec_phishing": (0.5752, 0.0210)},
        "gatv2":     {"macro_f1": (0.7281, 0.0116), "pr_auc": (0.8403, 0.0092), "rec_phishing": (0.4764, 0.0286)},
        "peaegnn":   {"macro_f1": (0.7534, 0.0431), "pr_auc": (0.8471, 0.0060), "rec_phishing": (0.5202, 0.0842)},
    }
    for m in ["graphsage", "gatv2", "peaegnn"]:
        ephg = ephg_results[m]
        print(f"{m:<12s} {'EthPhishGraph-2026':<20s} "
              f"{ephg['macro_f1'][0]:.4f} ± {ephg['macro_f1'][1]:.4f}   "
              f"{ephg['pr_auc'][0]:.4f} ± {ephg['pr_auc'][1]:.4f}   "
              f"{ephg['rec_phishing'][0]:.4f} ± {ephg['rec_phishing'][1]:.4f}")
        d2 = all_results[m]["agg"]
        print(f"{m:<12s} {'D2':<20s} "
              f"{d2['macro_f1']['mean']:.4f} ± {d2['macro_f1']['std']:.4f}   "
              f"{d2['pr_auc']['mean']:.4f} ± {d2['pr_auc']['std']:.4f}   "
              f"{d2['rec_phishing']['mean']:.4f} ± {d2['rec_phishing']['std']:.4f}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stage 8 - Cross-check on D2")
    parser.add_argument("--step", choices=["build", "train", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bs", type=int, default=64)
    args = parser.parse_args()

    if args.step in ("build", "all"):
        if D2_PKL.exists():
            print(f"[skip build] Da co {D2_PKL}. Xoa file de rebuild.")
        else:
            build_d2_dataset()
    if args.step in ("train", "all"):
        train_baselines_on_d2(epochs=args.epochs, seeds=tuple(args.seeds),
                              bs=args.bs, patience=args.patience)


if __name__ == "__main__":
    main()
