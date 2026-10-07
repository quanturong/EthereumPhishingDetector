"""
Stage 5 - Huan luyen baseline GNN tren EthPhishGraph-2026  (K=2)
================================================================
Models:
  - GraphSAGE  (Hamilton et al. 2017)  [13]
  - GATv2      (Brody et al. 2022)     [14]
  - PEAE-style (xap xi PEAE-GNN [1])   -- KHONG phai ban tai lap goc

Thay doi so voi ban K=1:
  1. Doc pkl K=2 (stage9_k2/ego_graph_k2_dataset.pkl).
  2. Bo feature thoi gian TUYET DOI:
       node: bo first_ts (cot 6), last_ts (cot 7); giu active_span (cot 8)
       edge: bo timestamp (cot 1); giu value, gas_price, gas_used
     Ly do: split theo thoi gian + ti le phishing lech theo partition
     (57% / 37% / 42%) -> mo hinh co the hoc "thoi diem -> nhan".
     LUU Y: gas_price cung la proxy cua thoi gian (gas thay doi theo nam).
  3. Thong ke chuan hoa tinh tich luy (khong concatenate toan bo train)
     va giai phong graph tho sau khi doi sang PyG -> do ton RAM.
  4. Luu checkpoint tung seed + feat_stats de Stage 6/7 dung lai.

Cach dung:
  python stage5_train_baselines.py --model graphsage --epochs 100
  python stage5_train_baselines.py --model all --seeds 42 43 44
  (het bo nho GPU: them --bs 32)
"""

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
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

# ── Duong dan ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\NCKH\Final")
STAGE9_DIR  = BASE_DIR / "ethphishgraph2026_stage9_k2"
STAGE5_DIR  = BASE_DIR / "ethphishgraph2026_stage5_baselines_k2"
DATASET_PKL = STAGE9_DIR / "ego_graph_k2_dataset.pkl"

# Node features goc (11 cot): in_count, out_count, in_value_sum, out_value_sum,
#   in_value_mean, out_value_mean, first_ts, last_ts, active_span,
#   unique_in_peers, unique_out_peers
NODE_COLS = [0, 1, 2, 3, 4, 5, 8, 9, 10]   # bo first_ts(6), last_ts(7)
# Edge attr goc (4 cot): value, timestamp, gas_price, gas_used
EDGE_COLS = [0, 2, 3]                       # bo timestamp(1)


# ── Chuan hoa feature ─────────────────────────────────────────────────────────
def _slog_np(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def _running_stats(graphs: list[dict]) -> dict:
    """Mean/std (population) tren log-feature, tinh tich luy, chi dung train."""
    nc, ec = len(NODE_COLS), len(EDGE_COLS)
    sx, sxx, nx = np.zeros(nc), np.zeros(nc), 0
    se, see, ne = np.zeros(ec), np.zeros(ec), 0
    for g in graphs:
        x = _slog_np(g["node_features"][:, NODE_COLS].astype(np.float64))
        e = _slog_np(g["edge_attr"][:, EDGE_COLS].astype(np.float64))
        sx += x.sum(0); sxx += (x ** 2).sum(0); nx += len(x)
        se += e.sum(0); see += (e ** 2).sum(0); ne += len(e)
    mx, me = sx / nx, se / ne
    return {
        "node_mean": mx, "node_std": np.sqrt(np.maximum(sxx / nx - mx ** 2, 0.0)),
        "edge_mean": me, "edge_std": np.sqrt(np.maximum(see / ne - me ** 2, 0.0)),
    }


def compute_feature_stats(graphs: list[dict]) -> dict:
    s = _running_stats(graphs)
    return {k: torch.tensor(v, dtype=torch.float) for k, v in s.items()}


def _log1p_signed(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def to_pyg(g: dict, feat_stats: dict) -> Data:
    x  = torch.from_numpy(g["node_features"][:, NODE_COLS]).float()
    ei = torch.from_numpy(g["edge_index"]).long()
    ea = torch.from_numpy(g["edge_attr"][:, EDGE_COLS]).float()

    x  = (_log1p_signed(x)  - feat_stats["node_mean"]) / (feat_stats["node_std"] + 1e-6)
    ea = (_log1p_signed(ea) - feat_stats["edge_mean"]) / (feat_stats["edge_std"] + 1e-6)

    target_mask = torch.zeros(x.size(0), dtype=torch.bool)
    target_mask[g["target_idx"]] = True

    return Data(
        x=x, edge_index=ei, edge_attr=ea,
        y=torch.tensor([g["label"]], dtype=torch.long),
        target_mask=target_mask,
    )


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
    Phien ban cu (giu lai de reproduce ket qua Stage 5-7 da chay):
      SAGE encoder + target emb + mean/max pool + feature dropout.
    """
    def __init__(self, in_dim, hid=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hid))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hid, hid))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hid * 3, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, 2),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        if self.training:
            x = F.dropout(x, p=0.1, training=True)
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        target_emb = x[data.target_mask]
        g_mean = global_mean_pool(x, batch)
        g_max  = global_max_pool(x,  batch)
        return self.head(torch.cat([target_emb, g_mean, g_max], dim=1))


class PEAEGNNv2(nn.Module):
    """
    PEAE-GNN v2 - gan voi paper goc hon [Huang et al. 2024]:
      1. Augmented Ego-Graph (AEG): random edge drop + feature mask (bao toan target).
      2. Target-centric attention aggregation (GAT-based encoder).
      3. Multi-scale target embedding: concat target_emb o tung layer.
      4. Target-as-query attention readout: target attends toan graph.
      5. Projection head cho contrastive loss (NT-Xent giua 2 augmented views).

    Call forward(data) -> logits (inference OR classification during train).
    Call forward(data, return_aug=True) -> (logits, proj1, proj2) de tinh contrastive loss.
    """
    def __init__(self, in_dim, hid=64, num_layers=2, heads=4, dropout=0.3,
                 edge_drop=0.2, feat_mask=0.15):
        super().__init__()
        from torch_geometric.utils import dropout_edge, softmax as geo_softmax
        from torch_geometric.nn import global_add_pool
        self._dropout_edge = dropout_edge
        self._geo_softmax = geo_softmax
        self._global_add_pool = global_add_pool

        self.heads = heads
        self.hid = hid
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_drop = edge_drop
        self.feat_mask = feat_mask
        self.enc_dim = hid * heads  # output dim after concat heads

        # Target-aware encoder: GATv2 (attention-based)
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hid, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(self.enc_dim, hid, heads=heads, concat=True, dropout=dropout))

        # Target-centric attention readout (target as query, all nodes as key/value)
        self.attn_q = nn.Linear(self.enc_dim, hid)
        self.attn_k = nn.Linear(self.enc_dim, hid)
        self.attn_v = nn.Linear(self.enc_dim, hid)

        # Readout: multi-scale target + mean + max + attention readout
        readout_dim = num_layers * self.enc_dim + 2 * self.enc_dim + hid
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid // 2, 2),
        )

        # Projection head cho contrastive loss
        self.proj = nn.Sequential(
            nn.Linear(self.enc_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid // 2),
        )

    def _augment(self, x, edge_index, target_mask):
        """AEG: random edge drop + feature mask (bao toan target node)."""
        mask = torch.bernoulli(torch.full_like(x, 1 - self.feat_mask))
        mask[target_mask] = 1.0
        x_aug = x * mask
        edge_index_aug, _ = self._dropout_edge(edge_index, p=self.edge_drop, training=True)
        return x_aug, edge_index_aug

    def _encode(self, x, edge_index):
        """Return (final_x, list of per-layer x)."""
        xs = []
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        return x, xs

    def _target_attn_readout(self, x, batch, target_mask):
        """Target-as-query attention: target attends over all nodes in its graph."""
        q_tgt = self.attn_q(x[target_mask])         # [B, hid]
        q = q_tgt[batch]                             # [N, hid] broadcast
        k = self.attn_k(x)                           # [N, hid]
        v = self.attn_v(x)                           # [N, hid]
        scores = (q * k).sum(dim=-1) / (self.hid ** 0.5)  # [N]
        attn = self._geo_softmax(scores, batch)      # [N]
        return self._global_add_pool(attn.unsqueeze(-1) * v, batch)  # [B, hid]

    def _forward_graph(self, x, ei, batch, tm):
        """Return readout vector + per-layer target embeddings."""
        x_final, xs = self._encode(x, ei)
        tgt_per_layer = [xl[tm] for xl in xs]        # list of [B, enc_dim]
        multi_scale_tgt = torch.cat(tgt_per_layer, dim=1)  # [B, L*enc_dim]
        g_mean = global_mean_pool(x_final, batch)
        g_max  = global_max_pool(x_final, batch)
        attn_r = self._target_attn_readout(x_final, batch, tm)
        g = torch.cat([multi_scale_tgt, g_mean, g_max, attn_r], dim=1)
        return g, tgt_per_layer[-1]

    def forward(self, data, return_aug=False):
        x, ei, batch, tm = data.x, data.edge_index, data.batch, data.target_mask
        g, tgt_final = self._forward_graph(x, ei, batch, tm)
        logits = self.head(g)
        if not return_aug or not self.training:
            return logits
        # AEG: two augmented views for contrastive loss
        x1, ei1 = self._augment(x, ei, tm)
        x2, ei2 = self._augment(x, ei, tm)
        _, tgt1 = self._forward_graph(x1, ei1, batch, tm)
        _, tgt2 = self._forward_graph(x2, ei2, batch, tm)
        proj1 = self.proj(tgt1)
        proj2 = self.proj(tgt2)
        return logits, proj1, proj2


def peae_contrastive_loss(proj1, proj2, temperature=0.5):
    """NT-Xent (SimCLR style): same-graph different-view = positive, else negative."""
    proj1 = F.normalize(proj1, dim=1)
    proj2 = F.normalize(proj2, dim=1)
    B = proj1.size(0)
    all_proj = torch.cat([proj1, proj2], dim=0)             # [2B, D]
    sim = (all_proj @ all_proj.T) / temperature             # [2B, 2B]
    mask_self = torch.eye(2 * B, device=sim.device).bool()
    sim = sim.masked_fill(mask_self, -1e9)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(sim.device)
    return F.cross_entropy(sim, labels)


MODELS = {
    "graphsage": GraphSAGEModel,
    "gatv2":     GATv2Model,
    "peaegnn":   PEAEGNNModel,     # phien ban cu, giu de backward-compat
    "peaegnnv2": PEAEGNNv2,        # phien ban moi, gan paper goc hon
}


# ── Train & eval ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch).cpu())
        all_y.append(batch.y.cpu())
    logits = torch.cat(all_logits, dim=0)
    y      = torch.cat(all_y, dim=0).numpy()
    probs  = F.softmax(logits, dim=1).numpy()
    pred   = probs.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "prec_normal":   float(prec[0]), "rec_normal":   float(rec[0]), "f1_normal":   float(f1[0]),
        "prec_phishing": float(prec[1]), "rec_phishing": float(rec[1]), "f1_phishing": float(f1[1]),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pr_auc":   float(average_precision_score(y, probs[:, 1])),
    }


def train_one(model_name: str, train_ds, val_ds, test_ds, in_dim: int,
              epochs=100, lr=1e-3, bs=64, patience=15, device="cpu", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MODELS[model_name](in_dim=in_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

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
    best_state, hist = None, []

    # Contrastive loss weight cho peaegnnv2 (AEG + NT-Xent).
    is_peae_v2 = (model_name == "peaegnnv2")
    alpha_contrastive = 0.1

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            if is_peae_v2:
                logits, proj1, proj2 = model(batch, return_aug=True)
                ce = F.cross_entropy(logits, batch.y, weight=cw)
                cont = peae_contrastive_loss(proj1, proj2)
                loss = ce + alpha_contrastive * cont
            else:
                loss = F.cross_entropy(model(batch), batch.y, weight=cw)
            loss.backward()
            opt.step()
            total_loss += float(loss) * batch.num_graphs
        train_loss = total_loss / len(train_ds)

        val_m = evaluate(model, val_loader, device)
        hist.append({"epoch": ep, "train_loss": train_loss, **val_m})

        if val_m["macro_f1"] > best_val:
            best_val, best_epoch, wait = val_m["macro_f1"], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  [ep {ep:3d}] {time.time()-t0:5.1f}s loss={train_loss:.4f}  "
                  f"val_f1={val_m['macro_f1']:.4f}  val_pr_auc={val_m['pr_auc']:.4f}  "
                  f"best={best_val:.4f}@{best_epoch}")
        if wait >= patience:
            print(f"  Early stop tai epoch {ep} (patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, device)   # test chi danh gia 1 lan
    print(f"  -> Test: macro_f1={test_m['macro_f1']:.4f}  "
          f"pr_auc={test_m['pr_auc']:.4f}  phish_recall={test_m['rec_phishing']:.4f}")
    return {"best_epoch": best_epoch, "best_val_f1": best_val,
            "test": test_m, "history": hist, "best_state": best_state}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 5 - GNN baseline training (K=2)")
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
    print(f"Node cols: {NODE_COLS} | Edge cols: {EDGE_COLS}")

    print(f"Loading {DATASET_PKL} ...")
    with open(DATASET_PKL, "rb") as f:
        raw = pickle.load(f)
    print(f"  train={len(raw['train'])}, val={len(raw['val'])}, test={len(raw['test'])}")

    # Thong ke chuan hoa chi tren train (chong ro ri)
    stats = compute_feature_stats(raw["train"])
    torch.save(stats, STAGE5_DIR / "feat_stats.pt")
    print(f"  Feature dims: node={stats['node_mean'].numel()}, edge={stats['edge_mean'].numel()}")

    # Doi sang PyG tung partition roi giai phong graph tho de do ton RAM
    print("Converting to PyG ...")
    ds = {}
    for part in ("train", "val", "test"):
        ds[part] = [to_pyg(g, stats) for g in raw[part]]
        raw[part] = None
        gc.collect()
    del raw
    gc.collect()

    for part, arr in ds.items():
        ys = np.array([g.y.item() for g in arr])
        print(f"  {part:5s}: {len(arr)} graphs, phishing={ys.mean():.1%}, "
              f"avg_nodes={np.mean([g.num_nodes for g in arr]):.0f}, "
              f"avg_edges={np.mean([g.num_edges for g in arr]):.0f}")
    in_dim = ds["train"][0].x.size(1)

    models = list(MODELS) if args.model == "all" else [args.model]
    all_results = {}

    for m in models:
        print(f"\n{'='*70}\n Model: {m.upper()}\n{'='*70}")
        seed_results = []
        for s in args.seeds:
            print(f"\n--- seed={s} ---")
            r = train_one(m, ds["train"], ds["val"], ds["test"], in_dim,
                          epochs=args.epochs, lr=args.lr, bs=args.bs,
                          patience=args.patience, device=device, seed=s)
            torch.save(r.pop("best_state"), STAGE5_DIR / f"model_{m}_seed{s}.pt")
            seed_results.append(r)

        agg = {}
        for k in seed_results[0]["test"]:
            vals = [r["test"][k] for r in seed_results]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        all_results[m] = {"per_seed": seed_results, "agg": agg}

        print(f"\n>>> {m.upper()} (n_seeds={len(args.seeds)}):")
        for k, v in agg.items():
            print(f"    {k:15s} {v['mean']:.4f} +- {v['std']:.4f}")

    out = STAGE5_DIR / f"results_{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        clean = {m: {"agg": r["agg"],
                     "per_seed": [{"best_epoch": s["best_epoch"],
                                   "best_val_f1": s["best_val_f1"],
                                   "test": s["test"]} for s in r["per_seed"]]}
                 for m, r in all_results.items()}
        json.dump(clean, f, indent=2)
    print(f"\nSaved -> {out}")

    print("\n" + "=" * 70 + "\n SUMMARY (mean +- std across seeds)\n" + "=" * 70)
    print(f"{'Model':<12s} {'MacroF1':<18s} {'PR-AUC':<18s} {'Phish Recall':<18s}")
    for m in models:
        a = all_results[m]["agg"]
        print(f"{m:<12s} "
              f"{a['macro_f1']['mean']:.4f} +- {a['macro_f1']['std']:.4f}  "
              f"{a['pr_auc']['mean']:.4f} +- {a['pr_auc']['std']:.4f}  "
              f"{a['rec_phishing']['mean']:.4f} +- {a['rec_phishing']['std']:.4f}")


if __name__ == "__main__":
    main()