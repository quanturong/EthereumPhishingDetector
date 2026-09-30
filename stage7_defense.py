"""
Stage 7 - Defense mechanisms
============================
Theo dong 101-120 de cuong (Noi dung 3-4).

Hai phuong phap phong thu:
  1. AdvTrain-SAGE       (baseline defense): SAGE + adversarial training
  2. Reliability-Aware   (RA-SAGE):          SAGE + edge reliability weighting
                                              + adversarial training + consistency loss

Reliability score r(e) in [0,1] cho moi canh, tinh tu:
  - Edge features (log-transformed value, timestamp, gas)
  - Node context (features cua src va dst)

Danh gia:
  - Clean test
  - Known attacks (TOH, TIH, BPH-T2P, BPH-P2T) - dung trong training
  - Unseen attacks (STC, MWR) - KHONG trong training, kiem tra khai quat hoa

Cach chay:
  python stage7_defense.py --method advtrain --epochs 60
  python stage7_defense.py --method ra       --epochs 60
  python stage7_defense.py --method both     --epochs 60
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, SAGEConv, global_max_pool, global_mean_pool
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stage3_pipeline import _build_one_ego_graph, PHISHING_HOP1_DIR
from stage5_train_baselines import (
    GraphSAGEModel, compute_feature_stats, evaluate, to_pyg,
)
from stage6_adversarial_attacks import (
    ATTACKS, TXS_PER_UNIT, _load_raw_txs, predict_probs,
)

# --- Paths -------------------------------------------------------------------
BASE_DIR    = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE3_DIR  = BASE_DIR / "ethphishgraph2026_stage3_dataset"
STAGE7_DIR  = BASE_DIR / "ethphishgraph2026_stage7_defense"
DATASET_PKL = STAGE3_DIR / "ego_graph_dataset.pkl"

KNOWN_ATTACKS  = ["TOH", "TIH", "BPH-T2P", "BPH-P2T"]
UNSEEN_ATTACKS = ["STC", "MWR"]


# --- Reliability-Aware SAGE Conv --------------------------------------------
class RASAGEConv(MessagePassing):
    """
    SAGE aggregator voi edge reliability weighting.
      r(e) = sigmoid(MLP([h_src, h_dst, edge_attr]))
      h'_v = W_self * h_v + W_msg * mean(r(e) * h_u for u in N(v))
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, hid: int = 32):
        super().__init__(aggr="mean")
        self.lin_msg  = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.rel_mlp  = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        self.last_r = None  # cache reliability scores for logging

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        rel_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        r = torch.sigmoid(self.rel_mlp(rel_input))     # [E, 1]
        self.last_r = r.detach()
        out = self.propagate(edge_index, x=x, r=r, edge_attr=edge_attr)
        return self.lin_self(x) + out

    def message(self, x_j, r):
        return r * self.lin_msg(x_j)


class RASAGEModel(nn.Module):
    """Reliability-Aware GraphSAGE."""
    def __init__(self, in_dim: int, edge_dim: int, hid: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RASAGEConv(in_dim, hid, edge_dim))
        for _ in range(num_layers - 1):
            self.convs.append(RASAGEConv(hid, hid, edge_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, 2),
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, ei, ea))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(g)


# --- Adversarial sample generator ---------------------------------------------
def gen_adv_sample(raw_txs: list[dict], target: str, label: int, T_end: int,
                   known_attacks: list[str], rng: np.random.Generator) -> dict | None:
    """Sinh 1 mau doi khang tu 1 target phishing. Random attack + random budget."""
    if label != 1:
        return None
    if not raw_txs:
        return None
    atk = rng.choice(known_attacks)
    budget = int(rng.choice([1, 3, 5, 10]))
    pert_txs = ATTACKS[atk](raw_txs, target, budget, rng)
    return _build_one_ego_graph(target, 1, T_end, pert_txs)


def build_adv_pool(train_raw_graphs: list[dict], T_end: int, rng: np.random.Generator,
                   n_per_epoch: int) -> list[dict]:
    """
    Sinh n_per_epoch mau adv tu tap phishing training.
    Goi 1 lan moi epoch de train mo hinh voi adv examples da them.
    """
    phish_targets = [g for g in train_raw_graphs if g["label"] == 1]
    if not phish_targets:
        return []
    adv_samples = []
    attempts = 0
    while len(adv_samples) < n_per_epoch and attempts < n_per_epoch * 3:
        g = phish_targets[int(rng.integers(0, len(phish_targets)))]
        raw = _load_raw_txs(g["target"])
        adv = gen_adv_sample(raw, g["target"], 1, T_end, KNOWN_ATTACKS, rng)
        if adv is not None:
            adv_samples.append(adv)
        attempts += 1
    return adv_samples


# --- Training with adversarial samples + consistency -------------------------
def kl_consistency(logits_clean, logits_adv):
    """KL(clean || adv) ep prediction cua adv gan clean."""
    p_clean = F.softmax(logits_clean, dim=1).detach()
    log_p_adv = F.log_softmax(logits_adv, dim=1)
    return F.kl_div(log_p_adv, p_clean, reduction="batchmean")


def train_defense(method: str, model, train_raw_graphs, train_ds, val_ds,
                  feat_stats, T_end, device, epochs=60, lr=1e-3, bs=64,
                  patience=12, adv_ratio=0.5, cons_weight=0.5, seed=42):
    """
    method: 'advtrain' or 'ra'
    - Moi epoch, sinh adv pool tu training phishing targets.
    - Mix clean + adv trong training batch.
    - Neu method='ra': them consistency loss KL(clean, adv) tren cung target.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_y = np.array([g.y.item() for g in train_ds])
    cw = torch.tensor(
        [len(train_y) / (2 * (train_y == 0).sum()),
         len(train_y) / (2 * (train_y == 1).sum())],
        dtype=torch.float,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    val_loader = DataLoader(val_ds, batch_size=bs)

    best_val, best_state, wait = -1.0, None, 0
    n_phish_train = int((train_y == 1).sum())
    n_adv_per_epoch = int(adv_ratio * n_phish_train)

    for ep in range(1, epochs + 1):
        t0 = time.time()

        # Sinh adv pool cho epoch nay
        adv_graphs = build_adv_pool(train_raw_graphs, T_end, rng, n_adv_per_epoch)
        adv_ds = [to_pyg(g, feat_stats) for g in adv_graphs]

        # Combine clean + adv
        combined = train_ds + adv_ds
        combined_loader = DataLoader(combined, batch_size=bs, shuffle=True)

        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in combined_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, weight=cw)

            # Consistency loss cho method='ra': lay 1 batch nho tu clean/adv paired
            if method == "ra" and adv_ds and len(adv_ds) >= 8:
                idx = rng.choice(len(adv_ds), size=min(8, len(adv_ds)), replace=False)
                adv_batch = next(iter(DataLoader([adv_ds[i] for i in idx], batch_size=8))).to(device)
                # Tim clean tuong ung: khong mapping thuan tien -> tao random clean batch
                cln_idx = rng.choice(len(train_ds), size=min(8, len(train_ds)), replace=False)
                cln_batch = next(iter(DataLoader([train_ds[i] for i in cln_idx], batch_size=8))).to(device)
                l_clean = model(cln_batch)
                l_adv   = model(adv_batch)
                loss = loss + cons_weight * kl_consistency(l_clean, l_adv)

            loss.backward()
            opt.step()
            total_loss += float(loss) * batch.num_graphs
            n_seen += batch.num_graphs
        train_loss = total_loss / max(n_seen, 1)

        # Val
        vm = evaluate(model, val_loader, device)
        if vm["macro_f1"] > best_val:
            best_val   = vm["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  [ep {ep:2d}] {time.time()-t0:5.1f}s  "
                  f"n_adv={len(adv_ds):4d}  loss={train_loss:.4f}  "
                  f"val_f1={vm['macro_f1']:.4f}  best={best_val:.4f}")
        if wait >= patience:
            print(f"  Early stop tai epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# --- Robustness evaluation across known + unseen attacks ---------------------
@torch.no_grad()
def eval_all_attacks(model, phish_test_graphs, attacks: list[str], budgets: list[int],
                     feat_stats, T_end, device, seed=42) -> dict:
    """Return per-attack per-budget ASR + robust_recall."""
    rng = np.random.default_rng(seed)
    clean_probs = predict_probs(model, phish_test_graphs, feat_stats, device)
    clean_pred  = (clean_probs >= 0.5).astype(int)
    correct_mask = clean_pred == 1
    n_correct = int(correct_mask.sum())

    out = {"clean_recall": float(n_correct / len(phish_test_graphs)),
           "n_correct_clean": n_correct, "attacks": {}}

    for atk in attacks:
        out["attacks"][atk] = []
        for b in budgets:
            perturbed = []
            for g in phish_test_graphs:
                raw = _load_raw_txs(g["target"])
                if not raw:
                    perturbed.append(g); continue
                pt = ATTACKS[atk](raw, g["target"], b, rng)
                pg = _build_one_ego_graph(g["target"], 1, T_end, pt)
                perturbed.append(pg if pg is not None else g)
            atk_probs = predict_probs(model, perturbed, feat_stats, device)
            atk_pred  = (atk_probs >= 0.5).astype(int)
            flipped   = correct_mask & (atk_pred == 0)
            asr           = float(flipped.sum() / max(n_correct, 1))
            robust_recall = float((atk_pred == 1).sum() / len(phish_test_graphs))
            out["attacks"][atk].append({
                "budget": b, "amt": b * TXS_PER_UNIT[atk],
                "asr": asr, "robust_recall": robust_recall,
                "n_flipped": int(flipped.sum()),
            })
            print(f"    [{atk:8s} b={b:3d}] ASR={asr:.3f}  rob_rec={robust_recall:.3f}")
    return out


# --- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 7 - Defense mechanisms")
    parser.add_argument("--method", choices=["advtrain", "ra", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--adv_ratio", type=float, default=0.5)
    parser.add_argument("--cons_weight", type=float, default=0.5)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    STAGE7_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    with open(DATASET_PKL, "rb") as f:
        raw = pickle.load(f)
    stats = compute_feature_stats(raw["train"])
    train_ds = [to_pyg(g, stats) for g in raw["train"]]
    val_ds   = [to_pyg(g, stats) for g in raw["val"]]
    test_ds  = [to_pyg(g, stats) for g in raw["test"]]
    in_dim   = train_ds[0].x.size(1)
    edge_dim = train_ds[0].edge_attr.size(1)
    print(f"in_dim={in_dim}, edge_dim={edge_dim}")

    split_df = pd.read_csv(STAGE3_DIR / "temporal_split.csv")
    T_end = int(split_df["anchor_ts"].max()) + 10 * 365 * 24 * 3600

    phish_test = [g for g in raw["test"] if g["label"] == 1]
    print(f"Phishing test: {len(phish_test)}")

    methods = ["advtrain", "ra"] if args.method == "both" else [args.method]
    all_results = {}

    for m in methods:
        print(f"\n{'='*70}\n Training defense: {m.upper()}\n{'='*70}")
        if m == "advtrain":
            model = GraphSAGEModel(in_dim=in_dim).to(device)
        else:
            model = RASAGEModel(in_dim=in_dim, edge_dim=edge_dim).to(device)

        model, best_val = train_defense(
            method=m, model=model, train_raw_graphs=raw["train"],
            train_ds=train_ds, val_ds=val_ds,
            feat_stats=stats, T_end=T_end, device=device,
            epochs=args.epochs, lr=args.lr, bs=args.bs, patience=args.patience,
            adv_ratio=args.adv_ratio, cons_weight=args.cons_weight, seed=args.seed,
        )
        ckpt = STAGE7_DIR / f"model_{m}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved {ckpt}")

        # Clean eval
        test_loader = DataLoader(test_ds, batch_size=args.bs)
        clean = evaluate(model, test_loader, device)
        print(f"\n  Clean test: macro_f1={clean['macro_f1']:.4f}  "
              f"pr_auc={clean['pr_auc']:.4f}  phish_recall={clean['rec_phishing']:.4f}")

        # Known attacks eval
        print(f"\n  === Known attacks (in training) ===")
        known_r = eval_all_attacks(model, phish_test, KNOWN_ATTACKS, args.budgets,
                                   stats, T_end, device, seed=args.seed)

        # Unseen attacks eval
        print(f"\n  === Unseen attacks (NOT in training) ===")
        unseen_r = eval_all_attacks(model, phish_test, UNSEEN_ATTACKS, args.budgets,
                                    stats, T_end, device, seed=args.seed)

        all_results[m] = {
            "clean": clean, "best_val": best_val,
            "known":  known_r["attacks"],
            "unseen": unseen_r["attacks"],
            "clean_recall": known_r["clean_recall"],
            "n_correct_clean": known_r["n_correct_clean"],
        }

    out = STAGE7_DIR / f"results_{args.method}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out}")

    # Summary table
    print(f"\n{'='*80}\n SUMMARY: mean ASR across budgets\n{'='*80}")
    print(f"{'Method':<12s} {'Group':<8s} {'TOH':<8s} {'TIH':<8s} {'BPH-T2P':<10s} {'BPH-P2T':<10s} {'STC':<8s} {'MWR':<8s}")
    for m in methods:
        r = all_results[m]
        for grp, atks in [("known", KNOWN_ATTACKS), ("unseen", UNSEEN_ATTACKS)]:
            row = f"{m:<12s} {grp:<8s}"
            for a in ["TOH", "TIH", "BPH-T2P", "BPH-P2T", "STC", "MWR"]:
                if a in r[grp]:
                    mean_asr = np.mean([b["asr"] for b in r[grp][a]])
                    row += f" {mean_asr:.3f}  "
                else:
                    row += f" {'--':<8s}"
            print(row)


if __name__ == "__main__":
    main()
