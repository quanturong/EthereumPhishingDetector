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
  Chi huan luyen PEAE v2 (khong chay lai 3 mo hinh cu):  --model peaegnnv2 --seeds 42 43 44
  Kaggle: python stage5_train_baselines.py --pkl /kaggle/input/<ten-dataset>/ego_graph_k2_dataset.pkl \\
            --out /kaggle/working/stage5_k2 --model peaegnnv2 --seeds 42 43 44
  (het bo nho GPU: them --bs 32)

  PEAE-GNN BAN GIONG BAI NHAT (model 'peae_paper'; dac trung rieng, PHAI dung --out thu muc RIENG):
    python stage5_train_baselines.py --model peae_paper --bs 32 --epochs 64 --seeds 42 43 44 \\
        --pkl <pkl> --out <thu muc stage5_peae>
    (theo bai Huang et al. 2024: dac trung tang cuong thu cong, SAGE trung binh 2 lop, an 128, readout RTM top-3,
     dau phan loai 1 lop Linear + dropout, Min-Max THUAN (mac dinh; them --peae_log de bat log1p truoc Min-Max),
     CO them self-loop nhu bai;
     'all' KHONG gom peae_paper vi mo hinh nay dung bo dac trung va thong ke khac)

  CAC DIEM VAN KHAC BAI CO CHU Y (minh bach cho bao cao):
    1. Chia tap theo thoi gian 60/20/20 (bai chia ngau nhien) -> chong ro ri thong tin tuong lai.
    2. Cross-entropy co trong so lop (bai khong de cap) -> xu ly imbalance 56.8/43.2 cua train.
    3. K=2 ego-graph: DUNG NHU BAI (bai: K = 2, l = 2).
    4. Cat 500 giao dich/hang xom cu nhat (bai khong cat).
    5. AFTS dien giai: khoang thoi gian giua cac giao dich LIEN TIEP cua target (bai mo ta mo ho).
    6. Dau phan loai (1 lop Linear + dropout 0.5) va dropout: bai KHONG mo ta (chi noi cac phuong phap dung cung bo phan loai).
    7. Khong lay mau lang gieng (bai co lay mau, khong neu kich thuoc); dung toan bo lang gieng.
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
from torch_geometric.utils import add_self_loops, remove_self_loops

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


# ── PEAE-GNN ban giong bai: dac trung tang cuong (Algorithm 1 trong bai) ──────
# 24 chieu cho MOI node: [Din, Dout, AFTO(6), AFTI(6), MTin, MTout, RTin, RTout, AFTS(6)]
#   - Node MUC TIEU: du 24 chieu.   - Node KHAC: chi Din, Dout (cac chieu con lai = 0), dung nhu bai de giam nhieu.
#   - AF = (max, min, mean, median, std, sum).  AFTO/AFTI: tren gia tri giao dich RA/VAO cua node muc tieu.
#   - MT = max/sum gia tri theo huong.  RT = so hang xom KHAC NHAU / bac (cuong do tuong tac) theo huong.
#   - AFTS: AF tren khoang thoi gian giua cac giao dich LIEN TIEP cua node muc tieu (cach doc cua nhom; bai noi
#     "khoang thoi gian giua node u va hang xom v" nhung khong neu ro).
#   - Chuan hoa Min-Max THUAN theo tung chieu, thong ke CHI tren train (dung nhu bai, Algorithm 1 dong 13).
#     Luu y: Min-Max BAT BIEN theo ty le nen doi don vi wei -> ETH khong doi ket qua; van de la DUOI NANG: vai gia tri
#     rat lon se nen phan lon gia tri ve gan 0. Neu can, --peae_log ap log1p cho cac chieu dem/gia tri/thoi gian
#     TRUOC Min-Max (khac bai; chi dung de doi chung do nhay).
PEAE_DIM = 24
PEAE_RATIO_COLS = (14, 15, 16, 17)   # MTin, MTout, RTin, RTout: ty le, khong lay log


def _agg6(a) -> list:
    if len(a) == 0:
        return [0.0] * 6
    a = np.asarray(a, dtype=np.float64)
    return [float(a.max()), float(a.min()), float(a.mean()), float(np.median(a)), float(a.std()), float(a.sum())]


def peae_features_raw(g: dict) -> np.ndarray:
    n = g["node_features"].shape[0]
    src, dst = g["edge_index"][0], g["edge_index"][1]
    ea = g["edge_attr"]
    val, ts = ea[:, 0].astype(np.float64), ea[:, 1].astype(np.float64)
    X = np.zeros((n, PEAE_DIM), dtype=np.float64)
    X[:, 0] = np.bincount(dst, minlength=n)     # Din: so giao dich vao
    X[:, 1] = np.bincount(src, minlength=n)     # Dout: so giao dich ra
    t = int(g["target_idx"])
    m_out, m_in = src == t, dst == t
    vo, vi = val[m_out], val[m_in]
    X[t, 2:8] = _agg6(vo)
    X[t, 8:14] = _agg6(vi)
    X[t, 14] = vi.max() / vi.sum() if len(vi) and vi.sum() > 0 else 0.0     # MTin
    X[t, 15] = vo.max() / vo.sum() if len(vo) and vo.sum() > 0 else 0.0     # MTout
    X[t, 16] = len(np.unique(src[m_in])) / len(vi) if len(vi) else 0.0      # RTin
    X[t, 17] = len(np.unique(dst[m_out])) / len(vo) if len(vo) else 0.0     # RTout
    tt = np.sort(ts[m_out | m_in])
    X[t, 18:24] = _agg6(np.diff(tt)) if len(tt) >= 2 else [0.0] * 6         # AFTS
    return X


def peae_transform(X: np.ndarray, log: bool = False) -> np.ndarray:
    """Mac dinh KHONG tien xu ly (Min-Max thuan nhu bai). log=True: log1p cho cac chieu khong phai ty le (bien the doi chung)."""
    Y = X.astype(np.float64, copy=True)
    if log:
        cols = [c for c in range(PEAE_DIM) if c not in PEAE_RATIO_COLS]
        Y[:, cols] = np.log1p(np.maximum(Y[:, cols], 0.0))
    return Y


def compute_peae_stats(graphs: list, log: bool = False) -> dict:
    """Min/Max theo chieu tren TRAIN. Them node_mean/node_std/edge_* gia de cac script khac (in_dim) van chay.
    Co 'peae_log' duoc luu cung thong ke de luc tan cong/danh gia dung dung phep bien doi."""
    lo, hi = np.full(PEAE_DIM, np.inf), np.full(PEAE_DIM, -np.inf)
    for g in graphs:
        Y = peae_transform(peae_features_raw(g), log)
        lo, hi = np.minimum(lo, Y.min(0)), np.maximum(hi, Y.max(0))
    st = {"peae_min": torch.tensor(lo, dtype=torch.float), "peae_max": torch.tensor(hi, dtype=torch.float),
          "peae_log": torch.tensor(int(log)),
          "node_mean": torch.zeros(PEAE_DIM), "node_std": torch.ones(PEAE_DIM),
          "edge_mean": torch.zeros(len(EDGE_COLS)), "edge_std": torch.ones(len(EDGE_COLS))}
    return add_peae_edge_stats(st, graphs)        # them thong ke canh that (cho cac mo hinh phong thu RA)


def add_peae_edge_stats(st: dict, graphs: list) -> dict:
    """Them mean/std cua dac trung CANH (log value, gas_price, gas_used; chi tren train) va co 'peae_edge_ok'.
    PEAEGNNPaper khong dung dac trung canh; chi cac mo hinh RA (Stage 7) can. Dung de nang cap thong ke cu ma
    KHONG can huan luyen lai."""
    s_ = _running_stats(graphs)
    out = dict(st)
    out["edge_mean"] = torch.tensor(s_["edge_mean"], dtype=torch.float)
    out["edge_std"] = torch.tensor(s_["edge_std"], dtype=torch.float)
    out["peae_edge_ok"] = torch.tensor(1)
    return out


def to_pyg_peae(g: dict, st: dict) -> Data:
    x = torch.from_numpy(peae_transform(peae_features_raw(g), bool(int(st.get("peae_log", 0))))).float()
    x = (x - st["peae_min"]) / (st["peae_max"] - st["peae_min"] + 1e-8)       # Min-Max (khong cat ngoai [0,1])
    ei = torch.from_numpy(g["edge_index"]).long()
    tm = torch.zeros(x.size(0), dtype=torch.bool)
    tm[g["target_idx"]] = True
    if "peae_edge_ok" in st:                    # co thong ke canh that -> edge_attr cho RA (PEAEGNNPaper bo qua)
        e = _slog_np(g["edge_attr"][:, EDGE_COLS].astype(np.float64))
        ea = (torch.from_numpy(e).float() - st["edge_mean"]) / (st["edge_std"] + 1e-6)
    else:
        ea = torch.zeros((ei.size(1), len(EDGE_COLS)))
    return Data(x=x, edge_index=ei, edge_attr=ea,
                y=torch.tensor([g["label"]], dtype=torch.long), target_mask=tm)


def to_pyg(g: dict, feat_stats: dict) -> Data:
    if "peae_min" in feat_stats:               # thong ke cua mo hinh peae_paper -> dung bo dac trung cua bai
        return to_pyg_peae(g, feat_stats)
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


def rtm_readout(x: torch.Tensor, batch: torch.Tensor, n: int = 3) -> torch.Tensor:
    """RTM (Readout Top-n Mean) cua bai: voi MOI chieu bieu dien, sap xep gia tri cac node giam dan, lay trung binh top-n.
    n = 1 la max pooling. Do thu theo tung do thi (batch PyG da sap xep theo do thi)."""
    B = int(batch.max().item()) + 1
    ptr = [0] + torch.cumsum(torch.bincount(batch, minlength=B), 0).tolist()
    out = []
    for b in range(B):
        xb = x[ptr[b]:ptr[b + 1]]
        out.append(xb.topk(min(n, xb.size(0)), dim=0).values.mean(dim=0))
    return torch.stack(out, dim=0)


class PEAEGNNPaper(nn.Module):
    """
    PEAE-GNN ban GIONG BAI NHAT [Huang et al., IEEE TCSS 2024]:
      - Dau vao: 24 dac trung tang cuong (xem peae_features_raw), Min-Max THUAN.
      - Bo ma hoa: GraphSAGE mean, 2 lop, an 128, CO them self-loop nhu bai ("we add a self-loop to each node"):
        node tu gop chinh no vao trung binh lang gieng, nen node khong co canh vao van co thong tin lang gieng = chinh no.
        (Khong tuong duong root weight: root weight chi cong them W_r H_v, khong dua node vao phep trung binh.)
      - Readout: RTM top-n mean (n = 3).
      - Dau phan loai: dropout + 1 lop Linear (bai KHONG mo ta chi tiet; day la lua chon toi thieu cua nhom).
      - Mat mat: cross-entropy (trong so lop lam o train_one de xu ly imbalance).
    """
    def __init__(self, in_dim=PEAE_DIM, hid=128, num_layers=2, topn=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hid, aggr="mean")] +
                                   [SAGEConv(hid, hid, aggr="mean") for _ in range(num_layers - 1)])
        self.topn = topn
        # Dau phan loai: dropout + 1 lop Linear (bai khong mo ta)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid, 2))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ei, _ = remove_self_loops(ei)                      # bai: them self-loop cho moi node
        ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        for conv in self.convs:
            x = F.relu(conv(x, ei))
        return self.head(rtm_readout(x, batch, self.topn))


MODELS = {
    "graphsage": GraphSAGEModel,
    "gatv2":     GATv2Model,
    "peaegnn":   PEAEGNNModel,     # phien ban cu, giu de backward-compat
    "peaegnnv2": PEAEGNNv2,        # phien ban moi (KHONG giong bai: GATv2 + doi chieu)
    "peae_paper": PEAEGNNPaper,    # ban giong bai nhat (dac trung rieng, dung --out rieng)
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
    parser.add_argument("--pkl", default=None, help="duong dan ego_graph_k2_dataset.pkl (Kaggle/Colab)")
    parser.add_argument("--out", default=None, help="thu muc ghi ket qua (Kaggle: /kaggle/working/stage5_k2)")
    parser.add_argument("--peae_log", action="store_true",
                        help="CHI cho peae_paper: log1p truoc Min-Max (khac bai; mac dinh Min-Max thuan nhu bai)")
    args = parser.parse_args()

    global DATASET_PKL, STAGE5_DIR
    if args.pkl: DATASET_PKL = Path(args.pkl)
    if args.out: STAGE5_DIR = Path(args.out)

    STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Node cols: {NODE_COLS} | Edge cols: {EDGE_COLS}" if args.model != "peae_paper"
          else f"peae_paper: {PEAE_DIM} dac trung tang cuong (Din, Dout, AFTO, AFTI, MT, RT, AFTS), Min-Max theo train")

    print(f"Loading {DATASET_PKL} ...")
    with open(DATASET_PKL, "rb") as f:
        raw = pickle.load(f)
    print(f"  train={len(raw['train'])}, val={len(raw['val'])}, test={len(raw['test'])}")

    # Thong ke chuan hoa chi tren train (chong ro ri)
    use_peae = (args.model == "peae_paper")
    prev = STAGE5_DIR / "feat_stats.pt"
    if prev.exists():                        # khong de 2 bo thong ke (hai bo dac trung) ghi de len nhau
        old_st = torch.load(prev, map_location="cpu", weights_only=True)
        if use_peae and "peae_min" in old_st and int(old_st.get("peae_log", 0)) != int(args.peae_log):
            sys.exit(f"DUNG: {prev} dung peae_log={int(old_st.get('peae_log', 0))} nhung ban dang chay --peae_log={int(args.peae_log)}. "
                     f"Dung --out thu muc RIENG cho moi bien the (vd stage5_peae va stage5_peae_log).")
        if ("peae_min" in old_st) != use_peae:
            sys.exit(f"DUNG: {prev} la thong ke cua {'peae_paper' if 'peae_min' in old_st else 'cac mo hinh khac'}. "
                     f"Dung --out thu muc RIENG cho {'peae_paper' if use_peae else 'cac mo hinh khac'} de khong ghi de.")
    if args.peae_log and not use_peae:
        sys.exit("--peae_log chi dung voi --model peae_paper")
    stats = compute_peae_stats(raw["train"], args.peae_log) if use_peae else compute_feature_stats(raw["train"])
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

    models = [m for m in MODELS if m != "peae_paper"] if args.model == "all" else [args.model]
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