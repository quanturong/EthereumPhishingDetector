"""
Chan doan do nhay dac trung (RQ2) - KHONG phai tan cong on-chain
================================================================
Voi cac mau phishing mo hinh phat hien DUNG, thay truc tiep gia tri dac trung (khong qua
giao dich) bang gia tri 'dien hinh cua normal' o tung phan cua graph, roi do ty le bi lat:

  - target (tat ca / tung nhom dac trung)
  - hop-1 (toan bo), hop-2 (toan bo), hop-1 + hop-2
  - DOI CHUNG: thay target bang gia tri dien hinh cua PHISHING (P phai gan nhu khong doi;
    neu P tut manh thi chinh viec thay the gay nhieu, khong doc duoc ket qua)

Dien giai:
  - Thay target ma P tut manh  -> mo hinh dua vao dac trung cua target -> tan cong chi dong vao
    target co the hieu qua neu hien thuc hoa duoc bang giao dich.
  - Thay target ma KHONG lat, nhung thay hop-1/hop-2 lat -> mo hinh dua vao boi canh hang xom;
    tan cong chi them canh quanh target kho tac dong.
  - DOI CHUNG CHO BOI CANH: thay hop-1/hop-2 bang MAU THUC (boc ngau nhien tung nut tu
    bo nut hop-1/hop-2 that cua cac graph NORMAL, roi tu cac graph PHISHING). Giu duoc su
    khac nhau giua cac hang xom, nen khong bi OOD nhu viec gan 1 vector hang so.
    Neu mau-normal lat nhieu hon han mau-phishing -> mo hinh thuc su doc 'do giong normal'
    cua hang xom. Neu ca hai deu lat nhieu -> viec thay the dang gay nhieu.
Gia tri dien hinh = median cua cac nut tuong ung trong cac graph NORMAL cua tap test
(chi de chan doan, khong dung de chon mo hinh/sieu tham so).

Chay:
  python diag_feature_sensitivity.py --model peaegnn --stage5_dir D:\\NCKH\\Final\\results\\stage5_k2
  (thu nhanh: them --max_targets 100)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import stage9_k2_upgrade as k2
import stage6_adversarial_attacks as s6
from stage5_train_baselines import MODELS

# Chi so cot trong node_features goc (11 cot)
GROUPS = {
    "counts (in/out count)": [0, 1],
    "value sums": [2, 3],
    "value means": [4, 5],
    "active_span": [8],
    "unique peers": [9, 10],
}
SUBSAMPLE_HOP2 = 200      # so nut hop-2 lay mau / graph de tinh median


def _with_nf(g, nf):
    h = dict(g)
    h["node_features"] = nf
    return h


def _masks(g, h1):
    addrs = g["addresses"]
    h1m = np.fromiter((a in h1 for a in addrs), dtype=bool, count=len(addrs))
    tm = np.zeros(len(addrs), dtype=bool); tm[g["target_idx"]] = True
    return h1m & ~tm, ~h1m & ~tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="peaegnn")
    ap.add_argument("--ckpt_seed", type=int, default=42)
    ap.add_argument("--max_targets", type=int, default=0)
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--stage5_dir", default=None)
    args = ap.parse_args()

    stage5_dir, stage6_dir = s6.STAGE5_DIR, s6.STAGE6_DIR
    if args.base_dir:
        base = Path(args.base_dir)
        s6._set_base(base)
        stage5_dir = base / "ethphishgraph2026_stage5_baselines_k2"
        stage6_dir = base / "ethphishgraph2026_stage6_attacks_k2"
    if args.stage5_dir:
        stage5_dir = Path(args.stage5_dir)
    stage6_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = torch.load(stage5_dir / "feat_stats.pt", map_location="cpu", weights_only=True)
    if "peae_min" in stats:
        sys.exit("diag_feature_sensitivity.py CHUA ho tro peae_paper: phep thu thay cot dac trung cu (node_features) "
                 "khong anh huong toi dac trung cua mo hinh nay (tinh tu danh sach canh). Can viet lai phep thu rieng.")
    model = MODELS[args.model](in_dim=int(stats["node_mean"].numel())).to(device)
    model.load_state_dict(torch.load(stage5_dir / f"model_{args.model}_seed{args.ckpt_seed}.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    print(f"Device: {device} | Model: {args.model} | ckpt seed: {args.ckpt_seed}")

    split = pd.read_csv(k2.SPLIT_CSV)
    t_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600
    label_of = dict(zip(split["address"].str.lower(), split["label"].astype(int)))
    test = split[split["partition"] == "test"].reset_index(drop=True)
    if args.max_targets:
        test = test.sample(n=min(args.max_targets, len(test)), random_state=0).reset_index(drop=True)
    print(f"Test targets: {len(test)} (phishing={int((test['label'] == 1).sum())})")

    rng = np.random.default_rng(0)
    ph_graphs, ph_masks = [], []
    ph_target_rows, nm_target_rows, nm_h1_rows, nm_h2_rows = [], [], [], []
    ph_h1_rows, ph_h2_rows = [], []
    t0 = time.time()
    for _, row in tqdm(test.iterrows(), total=len(test), unit="tgt"):
        addr, label = row["address"].lower(), int(row["label"])
        c = s6.TargetCache(addr, label, t_end, label_of)
        g = s6.build_graph(c)
        if g is None:
            continue
        h1m, h2m = _masks(g, c.h1)
        nf = g["node_features"]
        if label == 1:
            ph_graphs.append(g); ph_masks.append((h1m, h2m))
            ph_target_rows.append(nf[g["target_idx"]])
            if h1m.any():
                ph_h1_rows.append(nf[h1m])
            idx2p = np.where(h2m)[0]
            if len(idx2p):
                ph_h2_rows.append(nf[rng.choice(idx2p, size=min(SUBSAMPLE_HOP2, len(idx2p)), replace=False)])
        else:
            nm_target_rows.append(nf[g["target_idx"]])
            if h1m.any():
                nm_h1_rows.append(nf[h1m])
            idx2 = np.where(h2m)[0]
            if len(idx2):
                pick = rng.choice(idx2, size=min(SUBSAMPLE_HOP2, len(idx2)), replace=False)
                nm_h2_rows.append(nf[pick])
    print(f"Dung graph: {(time.time() - t0) / 60:.1f} phut | phishing={len(ph_graphs)} "
          f"normal(target)={len(nm_target_rows)}")

    def _median(rows, name):
        if not rows:
            print(f"CANH BAO: khong co mau de tinh median {name} -> dung 0")
            return np.zeros(11)
        return np.median(np.concatenate([np.atleast_2d(r) for r in rows]), axis=0)

    if not ph_graphs or not nm_target_rows:
        raise SystemExit("Can ca mau phishing va normal trong test (tang --max_targets).")
    ref_nm_target = _median(nm_target_rows, "target normal")
    ref_ph_target = _median(ph_target_rows, "target phishing")
    ref_h1 = _median(nm_h1_rows, "hop-1 normal")
    ref_h2 = _median(nm_h2_rows, "hop-2 normal")

    p0 = s6.predict_graphs(model, ph_graphs, stats, device)
    correct = p0 >= 0.5
    n_correct = int(correct.sum())
    print(f"Phishing phat hien dung (clean): {n_correct}/{len(ph_graphs)} | "
          f"P trung binh (nhom dung) = {p0[correct].mean():.3f}")

    def intervene_target(cols, ref):
        def f(g, m):
            nf = g["node_features"].copy()
            nf[g["target_idx"], cols] = ref[cols]
            return _with_nf(g, nf)
        return f

    all_cols = list(range(11))
    interventions = {}
    interventions["target -> normal (tat ca dac trung)"] = intervene_target(all_cols, ref_nm_target)
    for name, cols in GROUPS.items():
        interventions[f"target -> normal: {name}"] = intervene_target(cols, ref_nm_target)

    def ctx(use_h1, use_h2):
        def f(g, m):
            nf = g["node_features"].copy()
            if use_h1: nf[m[0]] = ref_h1
            if use_h2: nf[m[1]] = ref_h2
            return _with_nf(g, nf)
        return f
    interventions["hop-1 -> normal"] = ctx(True, False)
    interventions["hop-2 -> normal"] = ctx(False, True)
    interventions["hop-1 + hop-2 -> normal"] = ctx(True, True)
    interventions["[DOI CHUNG] target -> phishing dien hinh"] = intervene_target(all_cols, ref_ph_target)

    def _pool(rows):
        return np.concatenate([np.atleast_2d(r) for r in rows]) if rows else np.zeros((1, 11))
    pools = {"normal": (_pool(nm_h1_rows), _pool(nm_h2_rows)),
             "phishing": (_pool(ph_h1_rows), _pool(ph_h2_rows))}
    rng2 = np.random.default_rng(1)

    def transplant(src, use_h1, use_h2):
        p1, p2 = pools[src]
        def f(g, m):
            nf = g["node_features"].copy()
            if use_h1 and m[0].any(): nf[m[0]] = p1[rng2.integers(0, len(p1), size=int(m[0].sum()))]
            if use_h2 and m[1].any(): nf[m[1]] = p2[rng2.integers(0, len(p2), size=int(m[1].sum()))]
            return _with_nf(g, nf)
        return f
    for src, tag in (("normal", ""), ("phishing", "[DOI CHUNG] ")):
        interventions[f"{tag}hop-1 -> mau thuc tu {src.upper()}"] = transplant(src, True, False)
        interventions[f"{tag}hop-2 -> mau thuc tu {src.upper()}"] = transplant(src, False, True)
        interventions[f"{tag}hop-1+hop-2 -> mau thuc tu {src.upper()}"] = transplant(src, True, True)

    idx = np.where(correct)[0]
    results = {"model": args.model, "ckpt_seed": args.ckpt_seed, "n_phishing": len(ph_graphs),
               "n_correct_clean": n_correct, "mean_p_clean_correct": float(p0[correct].mean()),
               "interventions": {}}
    print(f"\n{'Can thiep (thay gia tri truc tiep, KHONG phai giao dich)':<62s}{'lat':>8s}{'P TB':>8s}{'dP':>8s}")
    for name, fn in interventions.items():
        graphs = [fn(ph_graphs[i], ph_masks[i]) for i in idx]
        p = s6.predict_graphs(model, graphs, stats, device)
        flip = float((p < 0.5).mean())
        results["interventions"][name] = {"flip_rate": flip, "mean_p": float(p.mean()),
                                          "mean_dp": float(p.mean() - p0[idx].mean())}
        print(f"{name:<62s}{flip:>8.3f}{p.mean():>8.3f}{p.mean() - p0[idx].mean():>+8.3f}")

    out = stage6_dir / f"diag_feature_{args.model}_seed{args.ckpt_seed}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
