"""
Stage 6 - Tan cong doi khang o muc giao dich  (K=2)
====================================================
Khac ban K=1:
  1. Graph sau tan cong duoc dung lai bang CHINH ham _build_k2_ego cua Stage 9c
     (cung quy tac hop-1/hop-2, cung cutoff) -> clean va attacked khong lech cau truc.
     Khong con dung _build_one_ego_graph (K=1).
  2. Vi phu do ke tan cong tao ra duoc coi nhu hop-1/hop-2 hop le: tx cua chung
     (do attack sinh ra) duoc dung lam "file tx" cua vi do khi dung graph.
  3. Khong can nap pkl 4 GB. Can: temporal_split.csv, hop1_txs, normal_hop1_txs,
     hop2_neighbor_txs, checkpoint + feat_stats cua Stage 5.
     (Tuy chon --verify N: doi chieu graph dung lai voi pkl cua 9c.)
  4. Lap theo TARGET (moi target nap du lieu 1 lan, dung tat ca bien the tan cong,
     du doan theo lo nho) -> RAM thap.
  5. gas_price cua tx gia lay mau tu lich su gas cua chinh target (truoc day co dinh
     20 gwei -> 'chu ky' de nhan ra). gas_used = 21000 (dung voi chuyen ETH thuong).
     Khoang value va thoi gian GIU NGUYEN nhu dinh nghia tan cong cu.
  6. Kiem tra hop le: ngan sach, timestamp > tx that cuoi, ts <= T_end, dau mut
     do ke tan cong kiem soat, value > 0. KHONG kiem tra so du / chi phi gas.
  7. Them Robust Macro-F1 (tan cong phishing, normal giu nguyen), AMT thuc te
     (so tx them) va AMT-to-flip (ngan sach nho nhat trong luoi da thu lam flip).

MWR (phuong an A - round-trip) trong graph K=2:
  target -> W1 -> W2 -> target. Dong tien duoc dinh tuyen qua 2 vi phu roi QUAY VE
  target (chi mat phi gas). Ban cu (target -> W1 -> W2 -> pert) bi K=2 cat mat tx
  cuoi nen trung cau truc voi BPH-T2P; ban round-trip thi nhin thay du trong K=2:
  moi budget them 2 node hop-1 (W1, W2), 3 canh (gom canh W1->W2 giua 2 node hop-1),
  khong them hop-2; target vua nhan vua gui. Khac TOH/TIH/BPH/STC ve cau truc.
  Can cap nhat Hinh 1 va mo ta MWR trong de cuong cho khop.

Chay (tren may co D:\\NCKH\\Final):
  python stage6_adversarial_attacks.py --model peaegnn --max_targets 30   # thu nhanh
  python stage6_adversarial_attacks.py --model peaegnn --ckpt_seed 42
"""

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Batch
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import stage9_k2_upgrade as k2
from stage5_train_baselines import MODELS, to_pyg

# --- Paths -------------------------------------------------------------------
BASE_DIR   = Path(r"D:\NCKH\Final")
STAGE5_DIR = BASE_DIR / "ethphishgraph2026_stage5_baselines_k2"
STAGE6_DIR = BASE_DIR / "ethphishgraph2026_stage6_attacks_k2"


def _set_base(base: Path):
    """Tro cac duong dan cua module stage9 ve thu muc goc moi."""
    k2.BASE_DIR = base
    k2.STAGE3_DIR = base / "ethphishgraph2026_stage3_dataset"
    k2.STAGE9_DIR = base / "ethphishgraph2026_stage9_k2"
    k2.HOP2_TXS_DIR = k2.STAGE9_DIR / "hop2_neighbor_txs"
    k2.PHISHING_HOP1_DIR = base / "ethphishgraph2026_stage2_normal_crawl" / "hop1_txs"
    k2.NORMAL_HOP1_DIR = k2.STAGE3_DIR / "normal_hop1_txs"
    k2.SPLIT_CSV = k2.STAGE3_DIR / "temporal_split.csv"
    k2.SPLIT_META = k2.STAGE3_DIR / "temporal_split_meta.json"
    k2.K2_DATASET = k2.STAGE9_DIR / "ego_graph_k2_dataset.pkl"


# --- Wallet & tx generation --------------------------------------------------
def _fake_wallet(seed_str: str, idx: int) -> str:
    h = hashlib.sha256(f"{seed_str}::atk::{idx}".encode()).hexdigest()
    return "0x" + h[:40]


def _mk_tx(sender, receiver, value_wei, ts, gas_price, gas_used=21_000) -> dict:
    return {
        "from": sender.lower(), "to": receiver.lower(),
        "value": str(value_wei), "timeStamp": str(ts),
        "gasPrice": str(gas_price), "gasUsed": str(gas_used), "isError": "0",
        "hash": "0x" + hashlib.sha256(
            f"{sender}{receiver}{value_wei}{ts}{gas_price}".encode()).hexdigest(),
    }


def _last_ts(raw_txs):
    ts = [int(t.get("timeStamp", 0)) for t in raw_txs if t.get("timeStamp")]
    return max(ts) if ts else int(time.time())


def _gas_pool(raw_txs):
    gp = [int(t.get("gasPrice", 0) or 0) for t in raw_txs]
    gp = [g for g in gp if g > 0]
    return np.array(gp, dtype=np.int64) if gp else None


def _gp(rng, pool):
    if pool is not None and len(pool):
        return int(rng.choice(pool))
    return int(10 ** rng.uniform(9, 11))   # 1-100 gwei


# --- Attacks: tra ve (new_txs, vi_do_ke_tan_cong_kiem_soat) -------------------
def attack_toh(raw, target, b, rng, pool, w0=0, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        peer = _fake_wallet(target, i + w0); wal.add(peer)
        val = int(rng.uniform(1e15, 1e16))
        ts = t0 + i * 600 + int(rng.integers(0, 300))
        new.append(_mk_tx(target, peer, val, ts, _gp(rng, pool)))
    return new, wal


def attack_tih(raw, target, b, rng, pool, w0=0, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        peer = _fake_wallet(target, i + 10_000 + w0); wal.add(peer)
        val = int(rng.uniform(1e15, 1e16))
        ts = t0 + i * 600 + int(rng.integers(0, 300))
        new.append(_mk_tx(peer, target, val, ts, _gp(rng, pool)))
    return new, wal


def attack_bph_t2p(raw, target, b, rng, pool, w0=0, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        w1 = _fake_wallet(target, i + 20_000 + w0); pert = _fake_wallet(target, i + 30_000 + w0)
        wal |= {w1, pert}
        val = int(rng.uniform(1e15, 1e16))
        ts1 = t0 + i * 1200
        ts2 = ts1 + 300 + int(rng.integers(0, 200))
        new.append(_mk_tx(target, w1, val, ts1, _gp(rng, pool)))
        new.append(_mk_tx(w1, pert, int(val * 0.98), ts2, _gp(rng, pool)))
    return new, wal


def attack_bph_p2t(raw, target, b, rng, pool, w0=0, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        pert = _fake_wallet(target, i + 40_000 + w0); w1 = _fake_wallet(target, i + 50_000 + w0)
        wal |= {w1, pert}
        val = int(rng.uniform(1e15, 1e16))
        ts1 = t0 + i * 1200
        ts2 = ts1 + 300 + int(rng.integers(0, 200))
        new.append(_mk_tx(pert, w1, val, ts1, _gp(rng, pool)))
        new.append(_mk_tx(w1, target, int(val * 0.98), ts2, _gp(rng, pool)))
    return new, wal


def attack_stc(raw, target, b, rng, pool, w0=0, hv=False, **_):
    """Split-and-Temporal Camouflage: b*4 tx nho target -> 1 vi camouflage, rai 30 ngay."""
    k, span = 4, 30 * 24 * 3600
    t0, new = _last_ts(raw) + 3600, []
    camo = _fake_wallet(target, 60_000 + w0)
    total = b * k
    for i in range(total):
        val = int(10 ** rng.uniform(16, 18)) if hv else int(rng.uniform(1e14, 5e14))
        ts = t0 + int((i / max(total - 1, 1)) * span)
        new.append(_mk_tx(target, camo, val, ts, _gp(rng, pool)))
    return new, {camo}


def attack_mwr(raw, target, b, rng, pool, w0=0, **_):
    """Multi-Wallet Routing (round-trip): target -> W1 -> W2 -> target (3 tx / budget)."""
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        w1 = _fake_wallet(target, i + 70_000 + w0)
        w2 = _fake_wallet(target, i + 80_000 + w0)
        wal |= {w1, w2}
        val = int(rng.uniform(1e15, 1e16))
        ts1 = t0 + i * 1800
        ts2 = ts1 + 300 + int(rng.integers(0, 200))
        ts3 = ts2 + 300 + int(rng.integers(0, 200))
        new.append(_mk_tx(target, w1, val, ts1, _gp(rng, pool)))
        new.append(_mk_tx(w1, w2, int(val * 0.98), ts2, _gp(rng, pool)))
        new.append(_mk_tx(w2, target, int(val * 0.96), ts3, _gp(rng, pool)))
    return new, wal


# --- Tan cong moi ------------------------------------------------------------
def attack_adaptive_stc(raw, target, b, rng, pool, w0=0, low_value=False, **_):
    """
    Adaptive-STC: b*6 tx nho, value log-uniform 0.01-1 ETH, rai 90 ngay, tu can bang chieu
    (target nhan nhieu hon gui -> them tx GUI DI; nguoc lai -> them tx NHAN VAO).
    """
    k, span = 6, 90 * 24 * 3600
    t0, new = _last_ts(raw) + 3600, []
    camo_in = _fake_wallet(target, 100_000 + w0)
    camo_out = _fake_wallet(target, 100_001 + w0)
    t_lc = target.lower()
    in_count = sum(1 for t in raw if (t.get("to") or "").lower() == t_lc)
    out_count = sum(1 for t in raw if (t.get("from") or "").lower() == t_lc)
    total = b * k
    for i in range(total):
        val = int(rng.uniform(1e14, 5e14)) if low_value else int(10 ** rng.uniform(16, 18))
        ts = t0 + int((i / max(total - 1, 1)) * span)
        if in_count > out_count:
            new.append(_mk_tx(target, camo_out, val, ts, _gp(rng, pool))); out_count += 1
        else:
            new.append(_mk_tx(camo_in, target, val, ts, _gp(rng, pool))); in_count += 1
    return new, {camo_in, camo_out}


def _combined_split(b):
    b_stc = int(round(0.5 * b)); b_bph = int(round(0.3 * b))
    return b - b_stc - b_bph, b_stc, b_bph          # (TOH, STC, BPH-T2P); co the = 0


def attack_combined(raw, target, b, rng, pool, w0=0, **_):
    """Ket hop: ~50% STC + ~30% BPH-T2P + ~20% TOH (tinh theo don vi ngan sach, co the = 0)."""
    b_toh, b_stc, b_bph = _combined_split(b)
    r = [np.random.default_rng(int(rng.integers(0, 2 ** 31))) for _ in range(3)]
    new, wal = [], set()
    for fn, bb, rr in ((attack_toh, b_toh, r[0]), (attack_stc, b_stc, r[1]), (attack_bph_t2p, b_bph, r[2])):
        if bb > 0:
            t, w = fn(raw, target, bb, rr, pool, w0=w0)
            new += t; wal |= w
    return new, wal


def attack_greedy_query(raw, target, b, rng, pool, model=None, stats=None, device="cpu",
                        cache=None, build_fn=None, n_cands=4, w0=0, **_):
    """
    GREEDY-QUERY (KHONG phai gradient): moi buoc thu n_cands motif ngau nhien (moi motif
    ngan sach 1, vi MOI buoc khac nhau), hoi MO HINH BI TAN CONG, giu motif lam P(phishing)
    giam nhieu nhat. Dung som khi (a) da lat (P<0.5) hoac (b) khong motif nao lam P giam.
    Phai danh gia RIENG voi tung mo hinh (ke tan cong thich ung theo mo hinh do).
    """
    if model is None or build_fn is None or cache is None or stats is None:
        raise ValueError("GREEDY-QUERY can model, stats, cache, build_fn")
    motifs = ["TOH", "TIH", "STC", "BPH-T2P", "BPH-P2T", "MWR"]

    def prob(txs):
        g = build_fn(cache, txs if txs else None)
        if g is None:
            return None
        batch = Batch.from_data_list([to_pyg(g, stats)]).to(device)
        with torch.no_grad():
            return float(F.softmax(model(batch), dim=1)[0, 1].item())

    acc, wal = [], set()
    cur = prob([])
    for step in range(b):
        if cur is None or cur < 0.5:
            break
        best = None
        for _ in range(n_cands):
            atk = motifs[int(rng.integers(0, len(motifs)))]
            ct, cw = ATTACKS[atk](raw + acc, target, 1,
                                  np.random.default_rng(int(rng.integers(0, 2 ** 31))),
                                  pool, w0=(step + 1) * 10)
            pv = prob(acc + ct)
            if pv is not None and (best is None or pv < best[0]):
                best = (pv, ct, cw)
        if best is None or best[0] >= cur:
            break
        cur, acc, wal = best[0], acc + best[1], wal | best[2]
    return acc, wal


def attack_stc_hv(raw, target, b, rng, pool, **kw):
    """STC y het ban goc (4 tx/don vi, 30 ngay, 1 vi, chi gui di) nhung gia tri LON nhu ADAPTIVE-STC."""
    return attack_stc(raw, target, b, rng, pool, hv=True, **kw)


def attack_adaptive_stc_lv(raw, target, b, rng, pool, **kw):
    """ADAPTIVE-STC y het (6 tx/don vi, 90 ngay, can bang chieu) nhung gia tri NHO nhu STC goc."""
    return attack_adaptive_stc(raw, target, b, rng, pool, low_value=True, **kw)


def _adaptive_stc_variant(raw, target, b_units, rng, pool, w0, params):
    """
    ADAPTIVE-STC voi tham so tuy chinh:
      params = (dir_bias, value_log_lo, value_log_hi, span_days, k_txs_per_unit)
    """
    dir_bias, v_lo, v_hi, span_days, k = params
    span = span_days * 24 * 3600
    t0, new = _last_ts(raw) + 3600, []
    camo_in = _fake_wallet(target, 100_000 + w0)
    camo_out = _fake_wallet(target, 100_001 + w0)
    t_lc = target.lower()
    in_count = sum(1 for t in raw if (t.get("to") or "").lower() == t_lc)
    out_count = sum(1 for t in raw if (t.get("from") or "").lower() == t_lc)
    total = int(b_units * k)
    for i in range(total):
        val = int(10 ** rng.uniform(v_lo, v_hi))
        ts = t0 + int((i / max(total - 1, 1)) * span)
        # dir_bias in [-1, 1]: -1 = chi incoming, +1 = chi outgoing, 0 = balance
        bias_prob = (dir_bias + 1) / 2  # map ve [0, 1]
        force_out = rng.random() < bias_prob
        want_out = force_out or (in_count > out_count and not force_out)
        if want_out:
            new.append(_mk_tx(target, camo_out, val, ts, _gp(rng, pool))); out_count += 1
        else:
            new.append(_mk_tx(camo_in, target, val, ts, _gp(rng, pool))); in_count += 1
    return new, {camo_in, camo_out}


def attack_defense_adaptive(raw, target, b, rng, pool, model=None, stats=None,
                             device="cpu", cache=None, build_fn=None,
                             n_cands=8, w0=0, **_):
    """
    DEFENSE-ADAPTIVE: dung ADAPTIVE-STC motif lam base, moi buoc search param space
    (dir_bias, value_scale, time_span) qua n_cands=8 candidates, pick candidate lam
    P(phishing) giam nhieu nhat. Khac GREEDY-QUERY o cho:
      1. Base motif: ADAPTIVE-STC (manh 50% ASR vs basic 2-3%)
      2. Search parameter: dir_bias, value_log range, time span
      3. Multi-scale budget: 1-3 units per step tuy giai doan
      4. Early stop khi P(phish) < 0.3

    Can model, stats, device, cache, build_fn. Dung voi mo hinh phong thu (RA-SAGE,
    AdvTrain) de do do ben truoc tan cong thich ung.
    """
    if model is None or build_fn is None or cache is None or stats is None:
        raise ValueError("DEFENSE-ADAPTIVE can model, stats, cache, build_fn")

    def prob(txs):
        g = build_fn(cache, txs if txs else None)
        if g is None:
            return None
        batch = Batch.from_data_list([to_pyg(g, stats)]).to(device)
        with torch.no_grad():
            return float(F.softmax(model(batch), dim=1)[0, 1].item())

    # Param space: (dir_bias, value_log_lo, value_log_hi, span_days, k)
    param_grid = [
        ( 0.0, 16, 18, 90, 6),   # ADAPTIVE-STC default
        ( 0.0, 14, 15, 90, 6),   # ADAPTIVE-STC-LV
        (-0.5, 16, 18, 90, 6),   # bias incoming
        (+0.5, 16, 18, 90, 6),   # bias outgoing
        ( 0.0, 16, 18, 30, 6),   # short span
        ( 0.0, 16, 18, 180, 8),  # long span, more txs
        ( 0.0, 15, 17, 60, 4),   # smaller motif
        ( 0.0, 17, 19, 90, 6),   # larger value
    ][:n_cands]

    acc, wal = [], set()
    cur = prob([])
    if cur is None:
        return [], set()

    # Chia budget: moi step = 1 unit (6 tx), stop khi het budget hoac flipped
    for step in range(b):
        if cur < 0.3:   # da flip rong rai, stop
            break
        best = None
        for ci, params in enumerate(param_grid):
            r_cand = np.random.default_rng(int(rng.integers(0, 2**31)))
            try:
                cand_txs, cand_wal = _adaptive_stc_variant(
                    raw + acc, target, 1, r_cand, pool,
                    w0=(step + 1) * 100 + ci, params=params,
                )
            except Exception:
                continue
            pv = prob(acc + cand_txs)
            if pv is not None and (best is None or pv < best[0]):
                best = (pv, cand_txs, cand_wal)
        if best is None or best[0] >= cur - 1e-4:
            break   # khong candidate nao giam P -> stop
        cur, acc, wal = best[0], acc + best[1], wal | best[2]
    return acc, wal


BASE_ATTACKS = ["TOH", "TIH", "BPH-T2P", "BPH-P2T", "STC", "MWR"]
ATTACKS = {
    "TOH": attack_toh, "TIH": attack_tih,
    "BPH-T2P": attack_bph_t2p, "BPH-P2T": attack_bph_p2t,
    "STC": attack_stc, "MWR": attack_mwr,
    "ADAPTIVE-STC": attack_adaptive_stc, "COMBINED": attack_combined,
    "GREEDY-QUERY": attack_greedy_query,
    "STC-HV": attack_stc_hv, "ADAPTIVE-STC-LV": attack_adaptive_stc_lv,   # ablation gia tri
    "DEFENSE-ADAPTIVE": attack_defense_adaptive,   # white-box adaptive to defense model
}
# Chi so co dinh cua tung tan cong (de seed ngau nhien KHONG doi khi doi thu tu/tap --attacks)
ATTACK_IDS = {n: i for i, n in enumerate(ATTACKS)}
TXS_PER_UNIT = {"TOH": 1, "TIH": 1, "BPH-T2P": 2, "BPH-P2T": 2, "STC": 4, "MWR": 3, "ADAPTIVE-STC": 6,
                "STC-HV": 4, "ADAPTIVE-STC-LV": 6,
                "DEFENSE-ADAPTIVE": 6}  # trung binh 6 tx/budget (ADAPTIVE-STC motif)


def _tx_cost_eth(new):
    """(tong gia tri chuyen, tong phi gas) cua cac tx gia, don vi ETH. Gia tri CONG DON qua moi tx
    (tien chay qua nhieu chang duoc dem nhieu lan) -> KHONG phai von toi thieu can co."""
    val = sum(int(t["value"]) for t in new) / 1e18
    gas = sum(int(t["gasPrice"]) * int(t["gasUsed"]) for t in new) / 1e18
    return val, gas


def expected_txs(atk, b):
    """So tx ky vong de kiem tra ngan sach; None = khong kiem tra (so tx bien thien)."""
    if atk == "COMBINED":
        t, s_, p_ = _combined_split(b)
        return t + 4 * s_ + 2 * p_
    if atk in ("GREEDY-QUERY", "DEFENSE-ADAPTIVE"):
        return None   # so tx bien thien theo early-stop
    return b * TXS_PER_UNIT[atk]


def validate_new_txs(new_txs, wallets, target, last_real_ts, t_end, n_expected):
    """Tra ve danh sach loi (rong = hop le). Khong kiem tra so du / chi phi gas."""
    errs = []
    if n_expected is not None and len(new_txs) != n_expected:
        errs.append("budget")
    ok_ends = {target} | wallets
    for t in new_txs:
        ts = int(t["timeStamp"])
        if ts <= last_real_ts: errs.append("ts<=last_real")
        if ts > t_end: errs.append("ts>t_end")
        if t["from"] not in ok_ends or t["to"] not in ok_ends: errs.append("uncontrolled_endpoint")
        if int(t["value"]) <= 0: errs.append("value<=0")
    return sorted(set(errs))


# --- Dung graph: dung chinh logic 9c ----------------------------------------
def _neighbor_txs(n, label_of):
    if k2.USE_TARGET_FILES and n in label_of:
        return k2._load_hop1(n, label_of[n])[:k2.MAX_TX_PER_NEIGHBOR]
    return k2._load_neighbor_txs(n)


class TargetCache:
    """Du lieu cua 1 target, nap 1 lan, dung cho moi bien the tan cong."""
    def __init__(self, addr, label, cutoff, label_of):
        self.addr, self.label, self.cutoff = addr, label, cutoff
        self.txs = k2._load_hop1(addr, label)
        self.h1 = k2._hop1_before_cutoff(addr, self.txs, cutoff)
        self.nmap = {}
        for n in self.h1:
            nt = _neighbor_txs(n, label_of)
            if nt:
                self.nmap[n] = nt
        self.pool = _gas_pool(self.txs)


def build_graph(c: TargetCache, new_txs=None):
    if not new_txs:
        return k2._build_k2_ego(c.addr, c.label, c.cutoff, c.h1, c.txs, c.nmap)
    all_target_txs = c.txs + new_txs
    h1 = k2._hop1_before_cutoff(c.addr, all_target_txs, c.cutoff)
    nmap = dict(c.nmap)
    for w in h1 - c.h1:   # vi phu hop-1: tx do attack sinh ra la "file tx" cua no
        nmap[w] = [t for t in new_txs if t["from"] == w or t["to"] == w]
    return k2._build_k2_ego(c.addr, c.label, c.cutoff, h1, all_target_txs, nmap)


@torch.no_grad()
def predict_graphs(model, graphs, stats, device, bs=16):
    """OOM-safe: catch RuntimeError, fallback bs/4 roi bs=1 cho batch problematic."""
    import gc
    model.eval()
    out = []
    i = 0
    while i < len(graphs):
        chunk = graphs[i:i + bs]
        try:
            batch = Batch.from_data_list([to_pyg(g, stats) for g in chunk]).to(device)
            probs = F.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            out.append(probs)
            i += bs
        except RuntimeError as e:
            if 'memory' not in str(e).lower():
                raise
            # OOM: free + fallback bs=1 cho chunk nay
            gc.collect()
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [OOM at i={i}, bs={bs}] fallback to bs=1 for {len(chunk)} graphs", flush=True)
            for g in chunk:
                try:
                    b1 = Batch.from_data_list([to_pyg(g, stats)]).to(device)
                    p = F.softmax(model(b1), dim=1)[:, 1].cpu().numpy()
                    out.append(p)
                except RuntimeError as e2:
                    if 'memory' not in str(e2).lower():
                        raise
                    print(f"  [SKIP graph (OOM even bs=1): {g.get('target', '?')[:10]}]", flush=True)
                    out.append(np.array([0.5]))  # neutral prob
            i += bs
    return np.concatenate(out)


def _atk_rng(seed, addr, atk_idx, b):
    return np.random.default_rng([seed, zlib.crc32(addr.encode()), atk_idx, b])


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 6 - Adversarial attacks (K=2)")
    ap.add_argument("--model", choices=list(MODELS), default="peaegnn")
    ap.add_argument("--ckpt_seed", type=int, default=42, help="seed cua checkpoint Stage 5")
    ap.add_argument("--budgets", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--attacks", nargs="+", default=BASE_ATTACKS, choices=list(ATTACKS) + ["all"],
                    help="mac dinh: 6 tan cong goc. GREEDY-QUERY rat cham (hoi mo hinh nhieu lan)")
    ap.add_argument("--greedy_cands", type=int, default=4, help="so motif thu moi buoc cua GREEDY-QUERY")
    ap.add_argument("--seed", type=int, default=42, help="seed sinh tan cong")
    ap.add_argument("--max_targets", type=int, default=0, help="0 = tat ca; >0 de chay thu")
    ap.add_argument("--verify", type=int, default=0, help="doi chieu N graph voi pkl 9c (nap pkl 4 GB)")
    ap.add_argument("--tag", default="", help="hau to ten file ket qua (tranh ghi de khi doi budget/attacks)")
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--stage5_dir", default=None)
    args = ap.parse_args()
    if "all" in args.attacks:
        args.attacks = list(ATTACKS)

    global STAGE5_DIR, STAGE6_DIR
    if args.base_dir:
        base = Path(args.base_dir)
        _set_base(base)
        STAGE5_DIR = base / "ethphishgraph2026_stage5_baselines_k2"
        STAGE6_DIR = base / "ethphishgraph2026_stage6_attacks_k2"
    if args.stage5_dir:
        STAGE5_DIR = Path(args.stage5_dir)
    STAGE6_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model} | ckpt seed: {args.ckpt_seed}")

    stats = torch.load(STAGE5_DIR / "feat_stats.pt", map_location="cpu", weights_only=True)
    in_dim = int(stats["node_mean"].numel())
    model = MODELS[args.model](in_dim=in_dim).to(device)
    model.load_state_dict(torch.load(STAGE5_DIR / f"model_{args.model}_seed{args.ckpt_seed}.pt",
                                     map_location=device, weights_only=True))
    model.eval()

    split = pd.read_csv(k2.SPLIT_CSV)
    t_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600     # = cutoff test cua 9c
    label_of = dict(zip(split["address"].str.lower(), split["label"].astype(int)))
    test = split[split["partition"] == "test"].reset_index(drop=True)
    if args.max_targets:   # lay mau NGAU NHIEN co dinh seed (head() co the lech theo thoi gian/nhan)
        test = test.sample(n=min(args.max_targets, len(test)), random_state=0).reset_index(drop=True)
    print(f"Test targets: {len(test)} (phishing={int((test['label']==1).sum())})")

    # --- doi chieu voi pkl (tuy chon) ---
    if args.verify:
        import pickle
        print(f"Verify: nap {k2.K2_DATASET} ...")
        with open(k2.K2_DATASET, "rb") as f:
            ref = pickle.load(f)["test"]
        bad = 0
        for g in ref[:args.verify]:
            c = TargetCache(g["target"], g["label"], t_end, label_of)
            r = build_graph(c)
            if (r["n_nodes"], r["n_edges"], r["n_hop1"]) != (g["n_nodes"], g["n_edges"], g["n_hop1"]):
                bad += 1
        del ref
        print(f"Verify: {args.verify - bad}/{args.verify} graph khop pkl 9c (can {args.verify}/{args.verify})")
        if bad:
            raise SystemExit("Graph dung lai KHAC pkl 9c -> dung lai, kiem tra duong dan/du lieu.")

    # --- vong lap theo target ---
    y_all, clean_prob = [], []
    phish_pos = {}          # chi so trong danh sach phishing
    atk_prob = {a: {b: [] for b in args.budgets} for a in args.attacks}
    atk_ntx = {a: {b: [] for b in args.budgets} for a in args.attacks}   # so tx THUC TE da them
    atk_cost = {a: {b: [] for b in args.budgets} for a in args.attacks}  # (value ETH, gas ETH)
    n_skipped, violations = 0, 0
    t_start = time.time()
    for _, row in tqdm(test.iterrows(), total=len(test), unit="tgt"):
        addr, label = row["address"].lower(), int(row["label"])
        c = TargetCache(addr, label, t_end, label_of)
        g0 = build_graph(c)
        if g0 is None:
            n_skipped += 1
            continue
        graphs, tags = [g0], [("clean", 0)]
        if label == 1:
            last_real = _last_ts(c.txs)
            ctx = {"model": model, "stats": stats, "device": device, "cache": c,
                   "build_fn": build_graph, "n_cands": args.greedy_cands}
            for atk in args.attacks:
                for b in args.budgets:
                    new, wal = ATTACKS[atk](c.txs, addr, b, _atk_rng(args.seed, addr, ATTACK_IDS[atk], b),
                                            c.pool, **ctx)
                    atk_ntx[atk][b].append(len(new))
                    atk_cost[atk][b].append(_tx_cost_eth(new))
                    errs = validate_new_txs(new, wal, addr, last_real, t_end, expected_txs(atk, b))
                    if errs:
                        violations += 1
                        print(f"  [INVALID] {addr[:10]} {atk} b={b}: {errs}")
                    pg = build_graph(c, new)
                    graphs.append(pg if pg is not None else g0)
                    tags.append((atk, b))
        probs = predict_graphs(model, graphs, stats, device)
        y_all.append(label)
        clean_prob.append(float(probs[0]))
        if label == 1:
            for (atk, b), p in zip(tags[1:], probs[1:]):
                atk_prob[atk][b].append(float(p))
    print(f"\nThoi gian: {(time.time()-t_start)/60:.1f} phut | bo qua: {n_skipped} | tx khong hop le: {violations}")

    # --- tong hop ---
    y = np.array(y_all)
    cp = np.array(clean_prob)
    cpred = (cp >= 0.5).astype(int)
    ph = y == 1
    n_ph = int(ph.sum())
    correct = cpred[ph] == 1
    n_correct = int(correct.sum())
    clean_macro = float(f1_score(y, cpred, average="macro", zero_division=0))
    print(f"Clean (graph dung lai): macro_f1={clean_macro:.4f} phishing_recall={n_correct/max(n_ph,1):.4f} "
          f"({n_correct}/{n_ph})")

    results = {"model": args.model, "ckpt_seed": args.ckpt_seed, "n_test": int(len(y)),
               "n_phishing": n_ph, "n_correct_clean": n_correct,
               "clean_macro_f1": clean_macro, "mean_p_phish_clean": float(cp[ph].mean()) if n_ph else None, "clean_phish_recall": n_correct / max(n_ph, 1),
               "n_skipped": n_skipped, "n_invalid_tx": violations, "attacks": {}}
    any_flip = np.zeros(n_ph, dtype=bool)   # lat boi IT NHAT 1 (tan cong, ngan sach)
    for atk in args.attacks:
        per_b, min_b = [], np.full(n_ph, np.inf)
        amt_first = np.full(n_ph, np.nan)
        for b in args.budgets:
            ap_ = np.array(atk_prob[atk][b]) >= 0.5
            flipped = correct & ~ap_
            any_flip |= flipped
            asr = float(flipped.sum() / max(n_correct, 1))
            pred = cpred.copy(); pred[ph] = ap_.astype(int)
            rmf1 = float(f1_score(y, pred, average="macro", zero_division=0))
            first = flipped & ~np.isfinite(min_b)
            ntx_b = np.array(atk_ntx[atk][b], dtype=float)
            amt_first[first] = ntx_b[first]
            min_b = np.where(first, b, min_b)
            p_att = np.array(atk_prob[atk][b])
            cst = np.array(atk_cost[atk][b], dtype=float).reshape(-1, 2)
            per_b.append({"budget": b, "amt": float(ntx_b.mean()) if len(ntx_b) else None, "asr": asr,
                          "mean_value_eth": float(cst[:, 0].mean()) if len(cst) else None,
                          "mean_gas_eth": float(cst[:, 1].mean()) if len(cst) else None,
                          "mean_p_phish_attacked": float(p_att.mean()) if len(p_att) else None,
                          "robust_recall": float(ap_.sum() / max(n_ph, 1)), "robust_macro_f1": rmf1,
                          "n_flipped": int(flipped.sum()), "n_correct_clean": n_correct})
        flipped_any = np.isfinite(min_b)
        amt_to_flip = float(np.nanmean(amt_first[flipped_any])) if flipped_any.any() else None
        results["attacks"][atk] = {"budgets": per_b, "frac_flipped_any_budget": float(flipped_any.sum() / max(n_correct, 1)),
                                   "mean_amt_to_flip": amt_to_flip}

    results["asr_any"] = float(any_flip.sum() / max(n_correct, 1))
    results["n_flipped_any"] = int(any_flip.sum())
    out = STAGE6_DIR / f"results_{args.model}_seed{args.ckpt_seed}{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*70}\n ASR theo tan cong & ngan sach ({args.model})\n{'='*70}")
    print(f"{'Attack':<10s}" + "".join(f"{'b=' + str(b):>9s}" for b in args.budgets) + f"{'AMT->flip':>12s}")
    for atk in args.attacks:
        r = results["attacks"][atk]
        amt = r["mean_amt_to_flip"]
        print(f"{atk:<10s}" + "".join(f"{x['asr']:>9.3f}" for x in r["budgets"])
              + f"{('%.1f' % amt) if amt is not None else '-':>12s}")
    print(f"\n{'='*70}\n P(phishing) trung binh tren {n_ph} mau phishing "
          f"(clean = {results['mean_p_phish_clean']:.3f})\n{'='*70}")
    print(f"{'Attack':<10s}" + "".join(f"{'b=' + str(b):>9s}" for b in args.budgets))
    for atk in args.attacks:
        print(f"{atk:<10s}" + "".join(f"{x['mean_p_phish_attacked']:>9.3f}"
                                      for x in results["attacks"][atk]["budgets"]))
    print(f"\n{'='*70}\n Chi phi trung binh / mau o ngan sach lon nhat (b={args.budgets[-1]})\n{'='*70}")
    print(f"{'Attack':<14s}{'so tx':>9s}{'value ETH':>12s}{'gas ETH':>11s}")
    for atk in args.attacks:
        x = results["attacks"][atk]["budgets"][-1]
        print(f"{atk:<14s}{x['amt']:>9.1f}{x['mean_value_eth']:>12.3f}{x['mean_gas_eth']:>11.4f}")
    print(f"\nASR_any (lat boi it nhat 1 tan cong/ngan sach): {results['asr_any']:.3f} "
          f"({results['n_flipped_any']}/{n_correct})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()