"""
Stage 6 - Adversarial attacks o muc giao dich (transaction-level evasion)
=========================================================================
Theo dong 89-100 de cuong. Threat model:
  - Ke tan cong kiem soat target phishing address + mot so vi phu
  - Chi them tx moi, khong xoa/sua tx da xac nhan
  - Ngan sach: so tx toi da (b)
  - Timestamp phai sau observation window cua target

Sau moi tan cong: build lai ego-graph tu tx set moi (khong sua truc tiep feature)

6 chien luoc:
  Baseline (Hide and Seek [2]):
    TOH:      target -> perturbation (b outgoing txs)
    TIH:      perturbation -> target (b incoming txs)
    BPH-T2P:  target -> W1 -> perturbation (b two-hop paths, 2b txs)
    BPH-P2T:  perturbation -> W1 -> target (b two-hop paths, 2b txs)
  Novel (nhom de xuat):
    STC:      Split-and-Temporal Camouflage
              (them k tx nho gia van chuyen thuong xuyen)
    MWR:      Multi-Wallet Routing
              (target -> W1 -> W2 -> perturbation, 3 hops, 3b txs)

Metrics (dong 99):
  - ASR (Attack Success Rate): ty le sample phish -> normal sau attack
  - AMT (Average Modified Transactions): so tx them trung binh
  - Robust Recall / F1 theo budget
  - Clean baseline de tham chieu
"""

import argparse
import hashlib
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

from stage3_pipeline import _build_one_ego_graph, PHISHING_HOP1_DIR
from stage5_train_baselines import (
    MODELS, compute_feature_stats, to_pyg, train_one, evaluate,
)

# --- Paths -------------------------------------------------------------------
BASE_DIR   = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE3_DIR = BASE_DIR / "ethphishgraph2026_stage3_dataset"
STAGE6_DIR = BASE_DIR / "ethphishgraph2026_stage6_attacks"
DATASET_PKL = STAGE3_DIR / "ego_graph_dataset.pkl"
MODEL_CKPT  = STAGE6_DIR / "model_graphsage.pt"


# --- Wallet & tx generation --------------------------------------------------
def _fake_wallet(seed_str: str, idx: int) -> str:
    """Generate deterministic fake attacker wallet (40 hex chars)."""
    h = hashlib.sha256(f"{seed_str}::atk::{idx}".encode()).hexdigest()
    return "0x" + h[:40]


def _mk_tx(sender: str, receiver: str, value_wei: int, ts: int,
           gas_price=20_000_000_000, gas_used=21_000) -> dict:
    """Build a fake transaction matching Etherscan V2 txlist format."""
    return {
        "from":      sender.lower(),
        "to":        receiver.lower(),
        "value":     str(value_wei),
        "timeStamp": str(ts),
        "gasPrice":  str(gas_price),
        "gasUsed":   str(gas_used),
        "isError":   "0",
        "hash":      "0x" + hashlib.sha256(
            f"{sender}{receiver}{value_wei}{ts}".encode()
        ).hexdigest(),
    }


def _last_ts(raw_txs: list[dict]) -> int:
    ts = [int(t.get("timeStamp", 0)) for t in raw_txs if t.get("timeStamp")]
    return max(ts) if ts else int(time.time())


# --- Attack implementations --------------------------------------------------
def attack_toh(raw_txs, target, budget, rng, **_):
    """target -> perturbation (b outgoing txs)."""
    t0 = _last_ts(raw_txs) + 3600
    new_txs = []
    for i in range(budget):
        peer = _fake_wallet(target, i)
        val  = int(rng.uniform(1e15, 1e16))  # 0.001-0.01 ETH
        ts   = t0 + i * 600 + int(rng.integers(0, 300))
        new_txs.append(_mk_tx(target, peer, val, ts))
    return raw_txs + new_txs


def attack_tih(raw_txs, target, budget, rng, **_):
    """perturbation -> target (b incoming txs)."""
    t0 = _last_ts(raw_txs) + 3600
    new_txs = []
    for i in range(budget):
        peer = _fake_wallet(target, i + 10_000)
        val  = int(rng.uniform(1e15, 1e16))
        ts   = t0 + i * 600 + int(rng.integers(0, 300))
        new_txs.append(_mk_tx(peer, target, val, ts))
    return raw_txs + new_txs


def attack_bph_t2p(raw_txs, target, budget, rng, **_):
    """target -> W1 -> perturbation (2b txs, 2-hop path)."""
    t0 = _last_ts(raw_txs) + 3600
    new_txs = []
    for i in range(budget):
        w1   = _fake_wallet(target, i + 20_000)
        pert = _fake_wallet(target, i + 30_000)
        val  = int(rng.uniform(1e15, 1e16))
        ts1  = t0 + i * 1200
        ts2  = ts1 + 300 + int(rng.integers(0, 200))  # must be after ts1
        new_txs.append(_mk_tx(target, w1, val, ts1))
        new_txs.append(_mk_tx(w1, pert, int(val * 0.98), ts2))  # 2% gas leak
    return raw_txs + new_txs


def attack_bph_p2t(raw_txs, target, budget, rng, **_):
    """perturbation -> W1 -> target (2b txs, 2-hop path)."""
    t0 = _last_ts(raw_txs) + 3600
    new_txs = []
    for i in range(budget):
        pert = _fake_wallet(target, i + 40_000)
        w1   = _fake_wallet(target, i + 50_000)
        val  = int(rng.uniform(1e15, 1e16))
        ts1  = t0 + i * 1200
        ts2  = ts1 + 300 + int(rng.integers(0, 200))
        new_txs.append(_mk_tx(pert, w1, val, ts1))
        new_txs.append(_mk_tx(w1, target, int(val * 0.98), ts2))
    return raw_txs + new_txs


def attack_stc(raw_txs, target, budget, rng, **_):
    """
    Split-and-Temporal Camouflage:
    Them b*k tx nho tu target ra 1 vi camouflage co dinh, giai deu qua thoi gian dai.
    Muc dich: pha loang pattern high-value / bursty cua phishing.
    """
    k = 4  # so tx nho moi budget unit
    t0 = _last_ts(raw_txs) + 3600
    time_span = 30 * 24 * 3600  # 30 ngay
    camo = _fake_wallet(target, 60_000)  # 1 vi camouflage duy nhat
    new_txs = []
    total = budget * k
    for i in range(total):
        val = int(rng.uniform(1e14, 5e14))  # 0.0001-0.0005 ETH (rat nho)
        ts  = t0 + int((i / max(total - 1, 1)) * time_span)
        new_txs.append(_mk_tx(target, camo, val, ts))
    return raw_txs + new_txs


def attack_mwr(raw_txs, target, budget, rng, **_):
    """
    Multi-Wallet Routing: target -> W1 -> W2 -> perturbation (3 hops, 3b txs).
    Che nhieu lop hon BPH.
    """
    t0 = _last_ts(raw_txs) + 3600
    new_txs = []
    for i in range(budget):
        w1   = _fake_wallet(target, i + 70_000)
        w2   = _fake_wallet(target, i + 80_000)
        pert = _fake_wallet(target, i + 90_000)
        val  = int(rng.uniform(1e15, 1e16))
        ts1  = t0 + i * 1800
        ts2  = ts1 + 300 + int(rng.integers(0, 200))
        ts3  = ts2 + 300 + int(rng.integers(0, 200))
        new_txs.append(_mk_tx(target, w1, val,               ts1))
        new_txs.append(_mk_tx(w1, w2,     int(val * 0.98),   ts2))
        new_txs.append(_mk_tx(w2, pert,   int(val * 0.96),   ts3))
    return raw_txs + new_txs


ATTACKS = {
    "TOH":     attack_toh,
    "TIH":     attack_tih,
    "BPH-T2P": attack_bph_t2p,
    "BPH-P2T": attack_bph_p2t,
    "STC":     attack_stc,
    "MWR":     attack_mwr,
}

# So tx moi budget unit (de tinh AMT)
TXS_PER_UNIT = {"TOH": 1, "TIH": 1, "BPH-T2P": 2, "BPH-P2T": 2, "STC": 4, "MWR": 3}


# --- Robustness evaluation ---------------------------------------------------
def _load_raw_txs(addr: str) -> list[dict]:
    f = PHISHING_HOP1_DIR / f"{addr}.json"
    if not f.exists() or f.stat().st_size <= 2:
        return []
    return json.loads(f.read_text(encoding="utf-8"))


@torch.no_grad()
def predict_probs(model, graphs, feat_stats, device, bs=64) -> np.ndarray:
    """Return P(phishing) for each ego-graph."""
    ds = [to_pyg(g, feat_stats) for g in graphs]
    loader = DataLoader(ds, batch_size=bs)
    probs = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        probs.append(F.softmax(logit, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def evaluate_attack(model, phish_test_graphs, attack_name, attack_fn,
                    budgets, feat_stats, device, T_end, threshold=0.5,
                    seed=42) -> dict:
    """
    ASR / AMT / robust recall theo budget.
    ASR = ty le sample bi flip tu phishing -> normal.
    AMT = so tx them trung binh cho toi flip (voi mau bi flip).
    Robust Recall = phish_recall tren tap test co bi tan cong.
    """
    rng = np.random.default_rng(seed)

    # Clean baseline: chi lay sample duoc predict dung la phishing
    clean_probs = predict_probs(model, phish_test_graphs, feat_stats, device)
    clean_pred  = (clean_probs >= threshold).astype(int)
    correct_mask = clean_pred == 1
    n_correct = int(correct_mask.sum())
    print(f"  Clean phishing detection: {n_correct}/{len(phish_test_graphs)} "
          f"({n_correct/len(phish_test_graphs):.3f} recall)")

    results = []
    for b in budgets:
        perturbed_graphs = []
        for g in phish_test_graphs:
            raw = _load_raw_txs(g["target"])
            if not raw:
                perturbed_graphs.append(g)  # fallback: no raw txs
                continue
            pert_txs = attack_fn(raw, g["target"], b, rng)
            pg = _build_one_ego_graph(g["target"], 1, T_end, pert_txs)
            perturbed_graphs.append(pg if pg is not None else g)

        atk_probs = predict_probs(model, perturbed_graphs, feat_stats, device)
        atk_pred  = (atk_probs >= threshold).astype(int)

        # ASR: chi tinh tren sample duoc predict dung ban dau
        flipped = (correct_mask) & (atk_pred == 0)
        asr = float(flipped.sum() / max(n_correct, 1))
        robust_recall = float((atk_pred == 1).sum() / len(phish_test_graphs))
        amt = b * TXS_PER_UNIT[attack_name]

        results.append({
            "budget": b, "amt": amt,
            "asr": asr, "robust_recall": robust_recall,
            "n_flipped": int(flipped.sum()), "n_correct_clean": n_correct,
        })
        print(f"  [b={b:3d}, AMT={amt:3d}] ASR={asr:.3f}  robust_recall={robust_recall:.3f}  "
              f"flipped={int(flipped.sum())}/{n_correct}")

    return {
        "attack":            attack_name,
        "clean_recall":      float(n_correct / len(phish_test_graphs)),
        "n_phish_test":      len(phish_test_graphs),
        "n_correct_clean":   n_correct,
        "budgets":           results,
    }


# --- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 6 - Adversarial attacks")
    parser.add_argument("--model", choices=list(MODELS), default="graphsage")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS),
                        choices=list(ATTACKS) + ["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()
    if "all" in args.attacks:
        args.attacks = list(ATTACKS)

    STAGE6_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")

    # Load dataset
    print(f"Loading {DATASET_PKL}")
    with open(DATASET_PKL, "rb") as f:
        raw = pickle.load(f)
    stats = compute_feature_stats(raw["train"])
    train_ds = [to_pyg(g, stats) for g in raw["train"]]
    val_ds   = [to_pyg(g, stats) for g in raw["val"]]
    test_ds  = [to_pyg(g, stats) for g in raw["test"]]
    in_dim   = train_ds[0].x.size(1)

    # Train or load model
    ckpt = STAGE6_DIR / f"model_{args.model}.pt"
    if ckpt.exists() and not args.force_retrain:
        print(f"Loading model checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model = MODELS[args.model](in_dim=in_dim).to(device)
        model.load_state_dict(state)
    else:
        print("Training fresh model (1 seed, 100 epochs)...")
        model = MODELS[args.model](in_dim=in_dim).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        train_y = np.array([g.y.item() for g in train_ds])
        cw = torch.tensor([
            len(train_y) / (2 * (train_y == 0).sum()),
            len(train_y) / (2 * (train_y == 1).sum()),
        ], dtype=torch.float).to(device)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=64)
        best_val, best_state, wait = -1.0, None, 0
        for ep in range(1, 101):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(batch), batch.y, weight=cw)
                loss.backward(); opt.step()
            vm = evaluate(model, val_loader, device)
            if vm["macro_f1"] > best_val:
                best_val = vm["macro_f1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if ep % 10 == 0:
                print(f"  ep {ep:3d}: val_f1={vm['macro_f1']:.4f} best={best_val:.4f}")
            if wait >= 15:
                break
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt)
        print(f"Saved {ckpt}")

    # Clean baseline on test
    test_loader = DataLoader(test_ds, batch_size=64)
    clean = evaluate(model, test_loader, device)
    print(f"\nClean test: macro_f1={clean['macro_f1']:.4f}  "
          f"pr_auc={clean['pr_auc']:.4f}  phish_recall={clean['rec_phishing']:.4f}")

    # Phishing test graphs (nhung sample dung phia phishing)
    phish_test = [g for g in raw["test"] if g["label"] == 1]
    print(f"\nPhishing test targets: {len(phish_test)}")

    # T_end de rebuild ego-graph (no cutoff for test)
    split = pd.read_csv(STAGE3_DIR / "temporal_split.csv")
    T_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600

    # Run attacks
    all_results = {"clean_test": clean, "model": args.model, "attacks": {}}
    for atk in args.attacks:
        print(f"\n{'='*70}\n ATTACK: {atk}\n{'='*70}")
        r = evaluate_attack(model, phish_test, atk, ATTACKS[atk],
                            args.budgets, stats, device, T_end,
                            seed=args.seed)
        all_results["attacks"][atk] = r

    out = STAGE6_DIR / f"results_{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}\n SUMMARY - ASR by attack & budget ({args.model})\n{'='*70}")
    header = f"{'Attack':<10s}" + "".join([f"{'b=' + str(b):>10s}" for b in args.budgets])
    print(header)
    for atk in args.attacks:
        row = f"{atk:<10s}"
        for br in all_results["attacks"][atk]["budgets"]:
            row += f"{br['asr']:>10.3f}"
        print(row)

    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
