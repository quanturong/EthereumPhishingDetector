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
def attack_toh(raw, target, b, rng, pool, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        peer = _fake_wallet(target, i); wal.add(peer)
        val = int(rng.uniform(1e15, 1e16))
        ts = t0 + i * 600 + int(rng.integers(0, 300))
        new.append(_mk_tx(target, peer, val, ts, _gp(rng, pool)))
    return new, wal


def attack_tih(raw, target, b, rng, pool, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        peer = _fake_wallet(target, i + 10_000); wal.add(peer)
        val = int(rng.uniform(1e15, 1e16))
        ts = t0 + i * 600 + int(rng.integers(0, 300))
        new.append(_mk_tx(peer, target, val, ts, _gp(rng, pool)))
    return new, wal


def attack_bph_t2p(raw, target, b, rng, pool, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        w1 = _fake_wallet(target, i + 20_000); pert = _fake_wallet(target, i + 30_000)
        wal |= {w1, pert}
        val = int(rng.uniform(1e15, 1e16))
        ts1 = t0 + i * 1200
        ts2 = ts1 + 300 + int(rng.integers(0, 200))
        new.append(_mk_tx(target, w1, val, ts1, _gp(rng, pool)))
        new.append(_mk_tx(w1, pert, int(val * 0.98), ts2, _gp(rng, pool)))
    return new, wal


def attack_bph_p2t(raw, target, b, rng, pool, **_):
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        pert = _fake_wallet(target, i + 40_000); w1 = _fake_wallet(target, i + 50_000)
        wal |= {w1, pert}
        val = int(rng.uniform(1e15, 1e16))
        ts1 = t0 + i * 1200
        ts2 = ts1 + 300 + int(rng.integers(0, 200))
        new.append(_mk_tx(pert, w1, val, ts1, _gp(rng, pool)))
        new.append(_mk_tx(w1, target, int(val * 0.98), ts2, _gp(rng, pool)))
    return new, wal


def attack_stc(raw, target, b, rng, pool, **_):
    """Split-and-Temporal Camouflage: b*4 tx nho target -> 1 vi camouflage, rai 30 ngay."""
    k, span = 4, 30 * 24 * 3600
    t0, new = _last_ts(raw) + 3600, []
    camo = _fake_wallet(target, 60_000)
    total = b * k
    for i in range(total):
        val = int(rng.uniform(1e14, 5e14))
        ts = t0 + int((i / max(total - 1, 1)) * span)
        new.append(_mk_tx(target, camo, val, ts, _gp(rng, pool)))
    return new, {camo}


def attack_mwr(raw, target, b, rng, pool, **_):
    """Multi-Wallet Routing (round-trip): target -> W1 -> W2 -> target (3 tx / budget)."""
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    for i in range(b):
        w1 = _fake_wallet(target, i + 70_000)
        w2 = _fake_wallet(target, i + 80_000)
        wal |= {w1, w2}
        val = int(rng.uniform(1e15, 1e16))
        ts1 = t0 + i * 1800
        ts2 = ts1 + 300 + int(rng.integers(0, 200))
        ts3 = ts2 + 300 + int(rng.integers(0, 200))
        new.append(_mk_tx(target, w1, val, ts1, _gp(rng, pool)))
        new.append(_mk_tx(w1, w2, int(val * 0.98), ts2, _gp(rng, pool)))
        new.append(_mk_tx(w2, target, int(val * 0.96), ts3, _gp(rng, pool)))
    return new, wal


# --- Attacks moi (Huong C: Adaptive-STC, D: Combined) ------------------------
def attack_adaptive_stc(raw, target, b, rng, pool, **ctx):
    """
    Adaptive-STC: ca bang in/out, value theo phan bo normal (log-uniform 0.01-1 ETH),
    rai qua 90 ngay. Chen ca 2 chieu de day phan bo in/out cua target ve trung tam normal.
    """
    k, span = 6, 90 * 24 * 3600
    t0, new, wal = _last_ts(raw) + 3600, [], set()
    camo_in = _fake_wallet(target, 100_000)
    camo_out = _fake_wallet(target, 100_001)
    wal |= {camo_in, camo_out}

    t_lc = target.lower()
    in_count = sum(1 for t in raw if (t.get("to") or "").lower() == t_lc)
    out_count = sum(1 for t in raw if (t.get("from") or "").lower() == t_lc)

    total = b * k
    for i in range(total):
        val = int(10 ** rng.uniform(16, 18))     # 0.01-1 ETH log-uniform
        ts = t0 + int((i / max(total - 1, 1)) * span)
        if in_count > out_count:
            new.append(_mk_tx(target, camo_out, val, ts, _gp(rng, pool)))
            out_count += 1
        else:
            new.append(_mk_tx(camo_in, target, val, ts, _gp(rng, pool)))
            in_count += 1
    return new, wal


def attack_combined(raw, target, b, rng, pool, **ctx):
    """
    Multi-strategy: 50% STC + 30% BPH-T2P + 20% TOH trong cung ngan sach.
    Realistic attacker ket hop nhieu motif.
    """
    b_stc = max(1, b // 2)
    b_bph = max(1, (b * 3) // 10)
    b_toh = max(1, b - b_stc - b_bph)
    rng1, rng2, rng3 = (np.random.default_rng(int(rng.integers(0, 2**31))) for _ in range(3))

    toh_txs, toh_w = attack_toh(raw, target, b_toh, rng1, pool)
    stc_txs, stc_w = attack_stc(raw, target, b_stc, rng2, pool)
    bph_txs, bph_w = attack_bph_t2p(raw, target, b_bph, rng3, pool)

    return toh_txs + stc_txs + bph_txs, toh_w | stc_w | bph_w


def attack_gradient_greedy(raw, target, b, rng, pool, **ctx):
    """
    Huong B: greedy whitebox - moi buoc chon motif + budget=1 ma giam P(phishing) manh nhat.
    Can ctx = {model, stats, device, cache, build_fn}. Khong co ctx -> fallback COMBINED.
    """
    model = ctx.get("model")
    stats = ctx.get("stats")
    device = ctx.get("device", "cpu")
    cache = ctx.get("cache")
    build_fn = ctx.get("build_fn")
    if model is None or build_fn is None or cache is None:
        return attack_combined(raw, target, b, rng, pool)

    base_motifs = ["TOH", "TIH", "STC", "BPH-T2P", "BPH-P2T", "MWR"]
    accepted_txs = []
    accepted_wal = set()
    n_tried_per_step = 4   # so motif try moi buoc (speed-vs-strength)

    for step in range(b):
        best_txs, best_wal, best_p = None, None, 1.0
        for _ in range(n_tried_per_step):
            atk = base_motifs[int(rng.integers(0, len(base_motifs)))]
            try:
                cand_txs, cand_wal = ATTACKS[atk](
                    raw + accepted_txs, target, 1,
                    np.random.default_rng(int(rng.integers(0, 2**31))), pool,
                )
            except Exception:
                continue
            try:
                g = build_fn(cache, accepted_txs + cand_txs)
                if g is None:
                    continue
                pyg = to_pyg(g, stats)
                batch = Batch.from_data_list([pyg]).to(device)
                with torch.no_grad():
                    p = float(F.softmax(model(batch), dim=1)[0, 1].item())
            except Exception:
                continue
            if p < best_p:
                best_p, best_txs, best_wal = p, cand_txs, cand_wal
        if best_txs:
            accepted_txs.extend(best_txs)
            accepted_wal |= best_wal
    return accepted_txs, accepted_wal


ATTACKS = {
    "TOH": attack_toh, "TIH": attack_tih,
    "BPH-T2P": attack_bph_t2p, "BPH-P2T": attack_bph_p2t,
    "STC": attack_stc, "MWR": attack_mwr,
    "ADAPTIVE-STC": attack_adaptive_stc,
    "COMBINED":     attack_combined,
    "GRADIENT":     attack_gradient_greedy,
}
# TXS_PER_UNIT dung de tinh AMT (so tx thuc te) va validate budget.
# COMBINED/GRADIENT co so tx bien thien theo cach split -> set thanh 0 de validator bo qua check "budget".
TXS_PER_UNIT = {
    "TOH": 1, "TIH": 1, "BPH-T2P": 2, "BPH-P2T": 2, "STC": 4, "MWR": 3,
    "ADAPTIVE-STC": 6, "COMBINED": 0, "GRADIENT": 0,
}


def validate_new_txs(new_txs, wallets, target, last_real_ts, t_end, n_expected):
    """Tra ve danh sach loi (rong = hop le). Khong kiem tra so du / chi phi gas.
    n_expected = 0: bo qua kiem tra budget (dung cho COMBINED/GRADIENT - so tx bien thien)."""
    errs = []
    if n_expected > 0 and len(new_txs) != n_expected:
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
    model.eval()
    out = []
    for i in range(0, len(graphs), bs):
        batch = Batch.from_data_list([to_pyg(g, stats) for g in graphs[i:i + bs]]).to(device)
        out.append(F.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def _atk_rng(seed, addr, atk_idx, b):
    return np.random.default_rng([seed, zlib.crc32(addr.encode()), atk_idx, b])


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 6 - Adversarial attacks (K=2)")
    ap.add_argument("--model", choices=list(MODELS), default="peaegnn")
    ap.add_argument("--ckpt_seed", type=int, default=42, help="seed cua checkpoint Stage 5")
    ap.add_argument("--budgets", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=list(ATTACKS) + ["all"])
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
            atk_ctx = {"model": model, "stats": stats, "device": device,
                       "cache": c, "build_fn": build_graph}
            for ai, atk in enumerate(args.attacks):
                for b in args.budgets:
                    new, wal = ATTACKS[atk](c.txs, addr, b,
                                            _atk_rng(args.seed, addr, ai, b),
                                            c.pool, **atk_ctx)
                    expected_n = b * TXS_PER_UNIT[atk]  # 0 cho COMBINED/GRADIENT -> skip budget check
                    errs = validate_new_txs(new, wal, addr, last_real, t_end, expected_n)
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
        for b in args.budgets:
            ap_ = np.array(atk_prob[atk][b]) >= 0.5
            flipped = correct & ~ap_
            any_flip |= flipped
            asr = float(flipped.sum() / max(n_correct, 1))
            pred = cpred.copy(); pred[ph] = ap_.astype(int)
            rmf1 = float(f1_score(y, pred, average="macro", zero_division=0))
            min_b = np.where(flipped & (min_b == np.inf), b, min_b)
            p_att = np.array(atk_prob[atk][b])
            per_b.append({"budget": b, "amt": b * TXS_PER_UNIT[atk], "asr": asr,
                          "mean_p_phish_attacked": float(p_att.mean()) if len(p_att) else None,
                          "robust_recall": float(ap_.sum() / max(n_ph, 1)), "robust_macro_f1": rmf1,
                          "n_flipped": int(flipped.sum()), "n_correct_clean": n_correct})
        flipped_any = np.isfinite(min_b)
        amt_to_flip = float(np.mean(min_b[flipped_any]) * TXS_PER_UNIT[atk]) if flipped_any.any() else None
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
    print(f"\nASR_any (lat boi it nhat 1 tan cong/ngan sach): {results['asr_any']:.3f} "
          f"({results['n_flipped_any']}/{n_correct})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()