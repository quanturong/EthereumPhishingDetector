"""
Stage 7 - Phong thu (K=2)  -  chia 3 buoc vi 3 buoc can may khac nhau
=====================================================================
  make_pool : (may co du lieu tho, CPU)  sinh san TAP GIAO DICH DOI KHANG de huan luyen
  train     : (Kaggle / may co GPU)      huan luyen AdvTrain / RA-SAGE
  eval      : (may co du lieu tho, CPU)  danh gia clean + known + unseen attack

Phuong phap:
  advtrain  : GraphSAGE + adversarial training (duong co so)
  ra        : RA-SAGE = SAGE + trong so tin cay canh r(e) + adversarial training
              + consistency loss (so sanh dung CAP clean/adv cua cung 1 target)
  ra_noedge : doi chung (ablation) - nhu 'ra' nhung r(e) KHONG nhin edge_attr
              (de biet loi ich co den tu thong tin canh hay chi tu viec co co che r(e))

Nhom tan cong:
  KNOWN  = TOH, TIH, BPH-T2P, BPH-P2T   (dung de sinh mau huan luyen)
  UNSEEN = STC, MWR(round-trip)          (KHONG dung trong huan luyen)

Thay doi so voi ban K=1:
  - Mau doi khang sinh bang dung ham dung graph K=2 cua Stage 6/9c, voi CUTOFF CUA TRAIN (T1),
    khong dung T_end (truoc day mau adv co lich su dai hon clean -> loi tat thoi gian).
    Tx goc dua cho tan cong bi cat <= cutoff de tx gia nam TRONG cua so quan sat.
  - Consistency loss ghep dung cap (clean, adv) cua CUNG target (truoc day ghep ngau nhien).
  - Tap adv sinh 1 lan (co dinh), khong dung lai graph moi epoch -> huan luyen nhanh tren GPU.
  - Dung lai feat_stats cua Stage 5 (cung chuan hoa de so sanh cong bang voi baseline).
  - eval dung 1 luot qua test cho TAT CA mo hinh (baseline, advtrain, ra, ra_noedge), luu xac
    suat tung mau (.npz) de sau nay kiem dinh ghep cap.

Cach chay:
  # 1) may co du lieu (~15-30 phut)
  python stage7_defense.py make_pool --stage5_dir D:\\NCKH\\Final\\results\\stage5_k2
  # 2) Kaggle (xem huong dan): train tung phuong phap
  python stage7_defense.py train --method advtrain --pkl <pkl> --pool <adv_pool_train.pkl> --stage5_dir <dir> --out <dir>
  # 3) may co du lieu
  python stage7_defense.py eval --stage5_dir D:\\NCKH\\Final\\results\\stage5_k2 --defense_dir <thu muc chua model_*.pt>
"""

import argparse
import gc
import json
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stage5_train_baselines import GraphSAGEModel, evaluate, to_pyg

BASE_DIR   = Path(r"D:\NCKH\Final")
STAGE5_DIR = BASE_DIR / "results" / "stage5_k2"
STAGE7_DIR = BASE_DIR / "ethphishgraph2026_stage7_defense_k2"

KNOWN_ATTACKS  = ["TOH", "TIH", "BPH-T2P", "BPH-P2T"]
UNSEEN_ATTACKS = ["STC", "MWR"]
ALL_ATTACKS    = KNOWN_ATTACKS + UNSEEN_ATTACKS
POOL_BUDGETS   = [1, 3, 5, 10]
CONS_METHODS   = {"ra", "ra_noedge"}


# --- RA-SAGE ------------------------------------------------------------------
class RASAGEConv(MessagePassing):
    """
    SAGE co trong so tin cay canh:
      r(e) = sigmoid(MLP([h_src, h_dst, edge_attr]))      (edge_dim=0: bo edge_attr)
      h'_v = W_self h_v + mean_{u in N(v)} r(e_uv) * W_msg h_u
    """
    def __init__(self, in_dim, out_dim, edge_dim, hid=32):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.rel_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.last_r = None

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        feats = [x[src], x[dst]]
        if self.edge_dim:
            feats.append(edge_attr)
        r = torch.sigmoid(self.rel_mlp(torch.cat(feats, dim=1)))   # [E, 1]
        self.last_r = r.detach()
        return self.lin_self(x) + self.propagate(edge_index, x=x, r=r)

    def message(self, x_j, r):
        return r * self.lin_msg(x_j)


class RASAGEModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hid=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([RASAGEConv(in_dim, hid, edge_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(RASAGEConv(hid, hid, edge_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 2))

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, ei, ea))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(g)


def make_model(method: str, in_dim: int, edge_dim: int):
    if method in ("baseline", "advtrain"):
        return GraphSAGEModel(in_dim=in_dim)
    if method == "ra":
        return RASAGEModel(in_dim, edge_dim)
    if method == "ra_noedge":
        return RASAGEModel(in_dim, 0)
    raise ValueError(method)


# =============================== make_pool ====================================
def cmd_make_pool(args):
    import pandas as pd
    import stage9_k2_upgrade as k2
    import stage6_adversarial_attacks as s6

    if args.base_dir:
        s6._set_base(Path(args.base_dir))
    out_dir = Path(args.out or STAGE7_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(k2.SPLIT_CSV)
    meta = json.loads(k2.SPLIT_META.read_text(encoding="utf-8"))
    t1 = int(meta["T1_ts"])
    label_of = dict(zip(split["address"].str.lower(), split["label"].astype(int)))
    train_ph = split[(split["partition"] == "train") & (split["label"] == 1)].reset_index(drop=True)
    if args.max_targets:
        train_ph = train_ph.head(args.max_targets)
    print(f"Train phishing targets: {len(train_ph)} | variants/target: {args.variants} | cutoff T1={t1}")

    pool, skipped_invalid, skipped_none = [], 0, 0
    for _, row in tqdm(train_ph.iterrows(), total=len(train_ph), unit="tgt"):
        addr = row["address"].lower()
        c = s6.TargetCache(addr, 1, t1, label_of)
        raw_in = [t for t in c.txs if int(t.get("timeStamp", 0) or 0) <= t1]   # chi tx trong cua so
        if not raw_in:
            skipped_none += 1
            continue
        last_real = s6._last_ts(raw_in)
        for v in range(args.variants):
            rng = np.random.default_rng([args.seed, zlib.crc32(addr.encode()), v, 7])
            atk = str(rng.choice(KNOWN_ATTACKS))
            b = int(rng.choice(POOL_BUDGETS))
            new, wal = s6.ATTACKS[atk](raw_in, addr, b, rng, c.pool)
            errs = s6.validate_new_txs(new, wal, addr, last_real, t1, b * s6.TXS_PER_UNIT[atk])
            if errs:                       # vd: tx gia tran qua cutoff T1
                skipped_invalid += 1
                continue
            g = s6.build_graph(c, new)
            if g is None:
                skipped_none += 1
                continue
            pool.append({
                "target": addr, "label": 1, "target_idx": int(g["target_idx"]),
                "node_features": g["node_features"].astype(np.float32),
                "edge_index": g["edge_index"].astype(np.int32),
                "edge_attr": g["edge_attr"].astype(np.float32),
                "attack": atk, "budget": b,
            })
    from collections import Counter
    print(f"Pool: {len(pool)} mau | bo qua (tx vuot cutoff): {skipped_invalid} | bo qua (khac): {skipped_none}")
    print("Theo tan cong:", dict(Counter(p["attack"] for p in pool)))
    out = out_dir / "adv_pool_train.pkl"
    with open(out, "wb") as f:
        pickle.dump({"pool": pool, "T1": t1, "known": KNOWN_ATTACKS, "seed": args.seed}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


# ================================ train =======================================
def kl_consistency(logits_clean, logits_adv):
    """KL(p_clean || p_adv); p_clean khong nhan gradient."""
    p_clean = F.softmax(logits_clean, dim=1).detach()
    return F.kl_div(F.log_softmax(logits_adv, dim=1), p_clean, reduction="batchmean")


def cmd_train(args):
    method = args.method
    out_dir = Path(args.out or STAGE7_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage5_dir = Path(args.stage5_dir or STAGE5_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device} | method: {method} | seed: {args.seed}")

    stats = torch.load(stage5_dir / "feat_stats.pt", map_location="cpu", weights_only=True)

    print(f"Loading {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        raw = pickle.load(f)
    train_ds = [to_pyg(g, stats) for g in raw["train"]]
    addr2idx = {g["target"]: i for i, g in enumerate(raw["train"])}
    val_ds = [to_pyg(g, stats) for g in raw["val"]]
    test_ds = [to_pyg(g, stats) for g in raw["test"]]
    del raw; gc.collect()
    for i, d in enumerate(train_ds):
        d.is_adv = torch.tensor([False]); d.pair_idx = torch.tensor([-1])

    with open(args.pool, "rb") as f:
        pool = pickle.load(f)["pool"]
    adv_ds = []
    for p in pool:
        if p["target"] not in addr2idx:
            continue
        d = to_pyg(p, stats)
        d.is_adv = torch.tensor([True]); d.pair_idx = torch.tensor([addr2idx[p["target"]]])
        adv_ds.append(d)
    del pool; gc.collect()
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} adv_pool={len(adv_ds)}")

    if args.smoke:    # chay thu nhanh: 300 train, 200 val, 1 epoch
        keep = set(rng.choice(len(train_ds), size=min(300, len(train_ds)), replace=False).tolist())
        remap = {old: new for new, old in enumerate(sorted(keep))}
        train_ds = [train_ds[i] for i in sorted(keep)]
        adv_ds = [d for d in adv_ds if int(d.pair_idx) in remap]
        for d in adv_ds: d.pair_idx = torch.tensor([remap[int(d.pair_idx)]])
        val_ds = val_ds[:200]; args.epochs = 1
        print(f"SMOKE: train={len(train_ds)} adv={len(adv_ds)} val={len(val_ds)}")

    in_dim = train_ds[0].x.size(1)
    edge_dim = train_ds[0].edge_attr.size(1)
    model = make_model(method, in_dim, edge_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    ty = np.array([d.y.item() for d in train_ds])
    cw = torch.tensor([len(ty) / (2 * (ty == 0).sum()), len(ty) / (2 * (ty == 1).sum())],
                      dtype=torch.float).to(device)
    val_loader = DataLoader(val_ds, batch_size=args.bs)
    n_adv_epoch = min(int(args.adv_ratio * int((ty == 1).sum())), len(adv_ds))
    use_cons = method in CONS_METHODS

    best_val, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        pick = rng.choice(len(adv_ds), size=n_adv_epoch, replace=False) if n_adv_epoch else []
        loader = DataLoader(train_ds + [adv_ds[i] for i in pick], batch_size=args.bs, shuffle=True)
        model.train()
        tot, seen = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, weight=cw)
            if use_cons:
                m = batch.is_adv.bool()
                if m.any():
                    ids = batch.pair_idx[m].tolist()          # chi so cac clean TUONG UNG
                    clean_b = Batch.from_data_list([train_ds[i] for i in ids]).to(device)
                    loss = loss + args.cons_weight * kl_consistency(model(clean_b), logits[m])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs
        vm = evaluate(model, val_loader, device)
        hist.append({"epoch": ep, "train_loss": tot / max(seen, 1), **vm})
        if vm["macro_f1"] > best_val:
            best_val, wait = vm["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  [ep {ep:3d}] {time.time() - t0:5.1f}s loss={tot / max(seen, 1):.4f} "
                  f"val_f1={vm['macro_f1']:.4f} best={best_val:.4f}")
        if wait >= args.patience:
            print(f"  Early stop tai epoch {ep}")
            break

    model.load_state_dict(best_state)
    tm = evaluate(model, DataLoader(test_ds, batch_size=args.bs), device)
    print(f"-> Clean test: macro_f1={tm['macro_f1']:.4f} pr_auc={tm['pr_auc']:.4f} "
          f"phish_recall={tm['rec_phishing']:.4f}")
    tag = f"{method}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    torch.save(model.state_dict(), out_dir / f"model_{tag}.pt")
    (out_dir / f"train_{tag}.json").write_text(
        json.dumps({"best_val_macro_f1": best_val, "clean_test": tm, "history": hist}, indent=2),
        encoding="utf-8")
    print(f"Saved -> {out_dir / f'model_{tag}.pt'}")


# ================================ eval ========================================
def summarize(y, cp, atk_probs, attacks, budgets, txs_per_unit):
    """y: nhan tat ca target test; cp: P(phishing) clean; atk_probs[atk][b]: P tung mau phishing."""
    y, cp = np.asarray(y), np.asarray(cp)
    cpred = (cp >= 0.5).astype(int)
    ph = y == 1
    n_ph = int(ph.sum())
    correct = cpred[ph] == 1
    n_correct = int(correct.sum())
    res = {"n_phishing": n_ph, "n_correct_clean": n_correct,
           "clean_macro_f1": float(f1_score(y, cpred, average="macro", zero_division=0)),
           "clean_phish_recall": n_correct / max(n_ph, 1), "attacks": {}}
    any_flip = np.zeros(n_ph, dtype=bool)
    for atk in attacks:
        rows = []
        for b in budgets:
            ap = np.asarray(atk_probs[atk][b]) >= 0.5
            flipped = correct & ~ap
            any_flip |= flipped
            pred = cpred.copy(); pred[ph] = ap.astype(int)
            rows.append({"budget": b, "amt": b * txs_per_unit[atk],
                         "asr": float(flipped.sum() / max(n_correct, 1)),
                         "n_flipped": int(flipped.sum()),
                         "robust_recall": float(ap.sum() / max(n_ph, 1)),
                         "robust_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))})
        res["attacks"][atk] = {"budgets": rows, "mean_asr": float(np.mean([r["asr"] for r in rows]))}
    res["asr_any"] = float(any_flip.sum() / max(n_correct, 1))
    res["n_flipped_any"] = int(any_flip.sum())
    for name, group in (("known", KNOWN_ATTACKS), ("unseen", UNSEEN_ATTACKS)):
        ms = [res["attacks"][a]["mean_asr"] for a in group if a in res["attacks"]]
        res[f"mean_asr_{name}"] = float(np.mean(ms)) if ms else None
    return res


def cmd_eval(args):
    import pandas as pd
    import stage9_k2_upgrade as k2
    import stage6_adversarial_attacks as s6

    stage5_dir = Path(args.stage5_dir or STAGE5_DIR)
    out_dir = Path(args.out or STAGE7_DIR)
    if args.base_dir:
        s6._set_base(Path(args.base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    defense_dir = Path(args.defense_dir) if args.defense_dir else out_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = torch.load(stage5_dir / "feat_stats.pt", map_location="cpu", weights_only=True)
    in_dim = int(stats["node_mean"].numel())
    edge_dim = int(stats["edge_mean"].numel())

    models = {}
    for m in args.methods:
        ck = (stage5_dir / f"model_graphsage_seed{args.seed}.pt") if m == "baseline" \
            else defense_dir / f"model_{m}_seed{args.seed}.pt"
        net = make_model(m, in_dim, edge_dim).to(device)
        net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        net.eval()
        models[m] = net
        print(f"Loaded {m}: {ck}")

    split = pd.read_csv(k2.SPLIT_CSV)
    t_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600
    label_of = dict(zip(split["address"].str.lower(), split["label"].astype(int)))
    test = split[split["partition"] == "test"].reset_index(drop=True)
    if args.max_targets:
        test = test.sample(n=min(args.max_targets, len(test)), random_state=0).reset_index(drop=True)
    print(f"Test targets: {len(test)} (phishing={int((test['label'] == 1).sum())})")

    ys = []
    cp = {m: [] for m in models}
    ap = {m: {a: {b: [] for b in args.budgets} for a in ALL_ATTACKS} for m in models}
    invalid = 0
    t0 = time.time()
    for _, row in tqdm(test.iterrows(), total=len(test), unit="tgt"):
        addr, label = row["address"].lower(), int(row["label"])
        c = s6.TargetCache(addr, label, t_end, label_of)
        g0 = s6.build_graph(c)
        if g0 is None:
            continue
        graphs, tags = [g0], [("clean", 0)]
        if label == 1:
            last_real = s6._last_ts(c.txs)
            for ai, atk in enumerate(ALL_ATTACKS):
                for b in args.budgets:
                    new, wal = s6.ATTACKS[atk](c.txs, addr, b, s6._atk_rng(args.attack_seed, addr, ai, b), c.pool)
                    if s6.validate_new_txs(new, wal, addr, last_real, t_end, b * s6.TXS_PER_UNIT[atk]):
                        invalid += 1
                    pg = s6.build_graph(c, new)
                    graphs.append(pg if pg is not None else g0)
                    tags.append((atk, b))
        ys.append(label)
        for m, net in models.items():
            p = s6.predict_graphs(net, graphs, stats, device)
            cp[m].append(float(p[0]))
            if label == 1:
                for (atk, b), pv in zip(tags[1:], p[1:]):
                    ap[m][atk][b].append(float(pv))
    print(f"\nThoi gian: {(time.time() - t0) / 60:.1f} phut | tx khong hop le: {invalid}")

    results = {m: summarize(ys, cp[m], ap[m], ALL_ATTACKS, args.budgets, s6.TXS_PER_UNIT) for m in models}
    tag = f"seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    (out_dir / f"eval_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / f"eval_probs_{tag}.npz",
                        labels=np.array(ys),
                        **{f"{m}__clean": np.array(cp[m]) for m in models},
                        **{f"{m}__{a}__{b}": np.array(ap[m][a][b])
                           for m in models for a in ALL_ATTACKS for b in args.budgets})

    base = results.get("baseline")
    print(f"\n{'=' * 96}\n ASR trung binh theo ngan sach (thap hon = tot hon). Chi con {len(models)} mo hinh.\n{'=' * 96}")
    print(f"{'Mo hinh':<11s}{'cleanF1':>8s}{'recall':>8s} |" + "".join(f"{a:>9s}" for a in ALL_ATTACKS)
          + f" |{'known':>7s}{'unseen':>8s}{'ANY':>7s}")
    for m, r in results.items():
        row = f"{m:<11s}{r['clean_macro_f1']:>8.3f}{r['clean_phish_recall']:>8.3f} |"
        row += "".join(f"{r['attacks'][a]['mean_asr']:>9.3f}" for a in ALL_ATTACKS)
        row += f" |{r['mean_asr_known']:>7.3f}{r['mean_asr_unseen']:>8.3f}{r['asr_any']:>7.3f}"
        print(row)
    if base:
        print("\nThay doi so voi baseline (am = phong thu giam ASR; clean: am = mat hieu nang):")
        for m, r in results.items():
            if m == "baseline":
                continue
            print(f"  {m:<10s} dCleanF1={r['clean_macro_f1'] - base['clean_macro_f1']:+.3f}  "
                  f"dASR_known={r['mean_asr_known'] - base['mean_asr_known']:+.3f}  "
                  f"dASR_unseen={r['mean_asr_unseen'] - base['mean_asr_unseen']:+.3f}")
    print(f"\nSaved -> {out_dir / f'eval_{tag}.json'}")


# ================================ main ========================================
def main():
    ap = argparse.ArgumentParser(description="Stage 7 - Defense (K=2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make_pool", help="sinh tap tx doi khang de huan luyen (may co du lieu tho)")
    p.add_argument("--variants", type=int, default=2, help="so mau doi khang / target phishing train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_targets", type=int, default=0)
    p.add_argument("--out", default=None); p.add_argument("--base_dir", default=None)
    p.add_argument("--stage5_dir", default=None)

    t = sub.add_parser("train", help="huan luyen phong thu (Kaggle/GPU)")
    t.add_argument("--method", choices=["advtrain", "ra", "ra_noedge"], required=True)
    t.add_argument("--pkl", required=True); t.add_argument("--pool", required=True)
    t.add_argument("--stage5_dir", default=None); t.add_argument("--out", default=None)
    t.add_argument("--epochs", type=int, default=60); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--bs", type=int, default=64); t.add_argument("--patience", type=int, default=12)
    t.add_argument("--adv_ratio", type=float, default=0.5); t.add_argument("--cons_weight", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--smoke", action="store_true", help="chay thu nhanh 1 epoch tren tap nho")

    e = sub.add_parser("eval", help="danh gia clean + known + unseen (may co du lieu tho)")
    e.add_argument("--methods", nargs="+", default=["baseline", "advtrain", "ra"],
                   choices=["baseline", "advtrain", "ra", "ra_noedge"])
    e.add_argument("--seed", type=int, default=42, help="seed cua checkpoint")
    e.add_argument("--attack_seed", type=int, default=42)
    e.add_argument("--budgets", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    e.add_argument("--max_targets", type=int, default=0)
    e.add_argument("--stage5_dir", default=None); e.add_argument("--defense_dir", default=None)
    e.add_argument("--out", default=None); e.add_argument("--base_dir", default=None)
    e.add_argument("--tag", default="")

    args = ap.parse_args()
    {"make_pool": cmd_make_pool, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()