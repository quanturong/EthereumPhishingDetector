"""
Stage 5 – Huấn luyện baseline GNN models trên EthPhishGraph-2026
================================================================
Models (theo dòng 28 & 40 đề cương):
  - GraphSAGE  (Hamilton et al. 2017)  [13]
  - GATv2      (Brody et al. 2022)     [14]
  - PEAE-GNN   (Huang et al. 2024)     [1]  -- xấp xỉ SAGE + target-attention

Metrics: Precision, Recall, Macro-F1, PR-AUC (dòng 59)
Repeat: nhiều seed, báo cáo mean ± std (dòng 59)

Cách dùng:
  python stage5_train_baselines.py --model graphsage --epochs 100
  python stage5_train_baselines.py --model gatv2     --epochs 100
  python stage5_train_baselines.py --model peaegnn   --epochs 100
  python stage5_train_baselines.py --model all       --seeds 3 5 7
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_fscore_support,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATv2Conv, SAGEConv, global_max_pool, global_mean_pool,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE3_DIR  = BASE_DIR / "ethphishgraph2026_stage3_dataset"
STAGE5_DIR  = BASE_DIR / "ethphishgraph2026_stage5_baselines"
DATASET_PKL = STAGE3_DIR / "ego_graph_dataset.pkl"


# ── Data loading & feature normalization ──────────────────────────────────────
def to_pyg(g: dict, feat_stats: dict) -> Data:
    x  = torch.from_numpy(g["node_features"]).float()
    ei = torch.from_numpy(g["edge_index"]).long()
    ea = torch.from_numpy(g["edge_attr"]).float()

    # Log-transform + z-score chuẩn hoá node features
    x = _normalize_features(x, feat_stats)
    ea = _normalize_edge_attr(ea, feat_stats.get("edge_mean"), feat_stats.get("edge_std"))

    # Target node mask (dùng cho PEAE-GNN readout)
    target_mask = torch.zeros(x.size(0), dtype=torch.bool)
    target_mask[g["target_idx"]] = True

    return Data(
        x=x, edge_index=ei, edge_attr=ea,
        y=torch.tensor([g["label"]], dtype=torch.long),
        target_mask=target_mask,
    )


def _log1p_signed(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def _normalize_features(x: torch.Tensor, stats: dict) -> torch.Tensor:
    x = _log1p_signed(x)
    return (x - stats["node_mean"]) / (stats["node_std"] + 1e-6)


def _normalize_edge_attr(ea: torch.Tensor, m, s):
    ea = _log1p_signed(ea)
    return (ea - m) / (s + 1e-6)


def compute_feature_stats(graphs: list[dict]) -> dict:
    all_x = np.concatenate([g["node_features"] for g in graphs], axis=0)
    all_e = np.concatenate([g["edge_attr"]     for g in graphs], axis=0)
    all_x = np.sign(all_x) * np.log1p(np.abs(all_x))
    all_e = np.sign(all_e) * np.log1p(np.abs(all_e))
    return {
        "node_mean": torch.tensor(all_x.mean(axis=0), dtype=torch.float),
        "node_std":  torch.tensor(all_x.std(axis=0),  dtype=torch.float),
        "edge_mean": torch.tensor(all_e.mean(axis=0), dtype=torch.float),
        "edge_std":  torch.tensor(all_e.std(axis=0),  dtype=torch.float),
    }


# ── Models ────────────────────────────────────────────────────────────────────
class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hid=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hid))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hid, hid))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, 2),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(g)


class GATv2Model(nn.Module):
    def __init__(self, in_dim, hid=64, heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hid, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hid * heads, hid, heads=heads, dropout=dropout))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hid * heads * 2, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, 2),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(g)


class PEAEGNNModel(nn.Module):
    """
    Xấp xỉ PEAE-GNN [1]:
      - SAGE encoder trên ego-graph
      - Kết hợp target-node embedding + graph-level readout (mean+max)
      - Thay thế "augmentation" gốc bằng feature-level dropout khi train
    """
    def __init__(self, in_dim, hid=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hid))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hid, hid))
        self.dropout = dropout
        # Target emb + mean + max = 3*hid
        self.head = nn.Sequential(
            nn.Linear(hid * 3, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, 2),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        if self.training:
            x = F.dropout(x, p=0.1, training=True)  # feature-level augmentation
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        target_emb = x[data.target_mask]  # [B, hid]
        g_mean = global_mean_pool(x, batch)
        g_max  = global_max_pool(x,  batch)
        g = torch.cat([target_emb, g_mean, g_max], dim=1)
        return self.head(g)


MODELS = {
    "graphsage": GraphSAGEModel,
    "gatv2":     GATv2Model,
    "peaegnn":   PEAEGNNModel,
}


# ── Train & eval ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_y.append(batch.y.cpu())
    logits = torch.cat(all_logits, dim=0)
    y      = torch.cat(all_y, dim=0).numpy()
    probs  = F.softmax(logits, dim=1).numpy()
    pred   = probs.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average=None, labels=[0, 1], zero_division=0)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    pr_auc   = average_precision_score(y, probs[:, 1])  # positive = phishing
    return {
        "prec_normal":   float(prec[0]), "rec_normal":   float(rec[0]), "f1_normal":   float(f1[0]),
        "prec_phishing": float(prec[1]), "rec_phishing": float(rec[1]), "f1_phishing": float(f1[1]),
        "macro_f1": float(macro_f1),
        "pr_auc":   float(pr_auc),
    }


def train_one(model_name: str, train_ds, val_ds, test_ds, in_dim: int,
              epochs=100, lr=1e-3, bs=64, patience=15, device="cpu", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MODELS[model_name](in_dim=in_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Class weight cho train imbalance
    train_y = np.array([g.y.item() for g in train_ds])
    cw = torch.tensor(
        [len(train_y) / (2 * (train_y == 0).sum()),
         len(train_y) / (2 * (train_y == 1).sum())],
        dtype=torch.float,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)
    test_loader  = DataLoader(test_ds,  batch_size=bs)

    best_val, best_epoch, wait = -1.0, 0, 0
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, weight=cw)
            loss.backward()
            opt.step()
            total_loss += float(loss) * batch.num_graphs
        train_loss = total_loss / len(train_ds)

        val_m = evaluate(model, val_loader, device)
        hist.append({"epoch": ep, "train_loss": train_loss, **val_m})

        if val_m["macro_f1"] > best_val:
            best_val   = val_m["macro_f1"]
            best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  [ep {ep:3d}] loss={train_loss:.4f}  val_f1={val_m['macro_f1']:.4f}  "
                  f"val_pr_auc={val_m['pr_auc']:.4f}  best={best_val:.4f}@{best_epoch}")
        if wait >= patience:
            print(f"  Early stop tại epoch {ep} (patience {patience})")
            break

    # Load best và evaluate test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, device)
    print(f"  → Test: macro_f1={test_m['macro_f1']:.4f}  "
          f"pr_auc={test_m['pr_auc']:.4f}  "
          f"phish_recall={test_m['rec_phishing']:.4f}")
    return {"best_epoch": best_epoch, "best_val_f1": best_val,
            "test": test_m, "history": hist}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 5 - GNN baseline training")
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--bs",     type=int,   default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seeds",  nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading {DATASET_PKL}...")
    with open(DATASET_PKL, "rb") as f:
        raw = pickle.load(f)
    print(f"  train={len(raw['train'])}, val={len(raw['val'])}, test={len(raw['test'])}")

    # Compute normalization stats trên train only (chống leak)
    stats = compute_feature_stats(raw["train"])
    print(f"  Feature dims: node={stats['node_mean'].numel()}, edge={stats['edge_mean'].numel()}")

    # Convert to PyG
    print("Converting to PyG format...")
    train_ds = [to_pyg(g, stats) for g in raw["train"]]
    val_ds   = [to_pyg(g, stats) for g in raw["val"]]
    test_ds  = [to_pyg(g, stats) for g in raw["test"]]
    in_dim = train_ds[0].x.size(1)

    models = list(MODELS) if args.model == "all" else [args.model]
    all_results = {}

    for m in models:
        print(f"\n{'='*70}\n Model: {m.upper()}\n{'='*70}")
        seed_results = []
        for s in args.seeds:
            print(f"\n--- seed={s} ---")
            r = train_one(m, train_ds, val_ds, test_ds, in_dim,
                          epochs=args.epochs, lr=args.lr, bs=args.bs,
                          patience=args.patience, device=device, seed=s)
            seed_results.append(r)

        # Aggregate
        agg = {}
        for k in seed_results[0]["test"]:
            vals = [r["test"][k] for r in seed_results]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        all_results[m] = {"per_seed": seed_results, "agg": agg}

        print(f"\n>>> {m.upper()} results (n_seeds={len(args.seeds)}):")
        for k, v in agg.items():
            print(f"    {k:15s} {v['mean']:.4f} ± {v['std']:.4f}")

    out = STAGE5_DIR / f"results_{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        # strip history for JSON size
        clean = {m: {"agg": r["agg"],
                     "per_seed": [{"best_epoch": s["best_epoch"],
                                   "best_val_f1": s["best_val_f1"],
                                   "test": s["test"]} for s in r["per_seed"]]}
                 for m, r in all_results.items()}
        json.dump(clean, f, indent=2)
    print(f"\nSaved -> {out}")

    print("\n" + "=" * 70)
    print(" SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    print(f"{'Model':<12s} {'MacroF1':<18s} {'PR-AUC':<18s} {'Phish Recall':<18s}")
    for m in models:
        a = all_results[m]["agg"]
        print(f"{m:<12s} "
              f"{a['macro_f1']['mean']:.4f} ± {a['macro_f1']['std']:.4f}  "
              f"{a['pr_auc']['mean']:.4f} ± {a['pr_auc']['std']:.4f}  "
              f"{a['rec_phishing']['mean']:.4f} ± {a['rec_phishing']['std']:.4f}")


if __name__ == "__main__":
    main()
