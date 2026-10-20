"""
Stage 8 (muc A) - Danh gia TRUC TIEP cac mo hinh da huan luyen tren du lieu 2026 len D2 (KHONG huan luyen lai).
================================================================================================
Dung du lieu da chuyen bang convert_d2.py (thu muc D2base). Graph K=2 duoc dung bang CHINH ham cua
Stage 9c/Stage 6; dac trung chuan hoa bang feat_stats.pt cua Stage 5 (thong ke cua tap train 2026).

  - Danh gia tren TOAN BO target D2 (lich su day du, khong cat theo thoi gian) vi mo hinh chua tung thay D2.
  - Bao cao them tren tap 'test' theo thoi gian cua D2 (neu co) de tham khao.
  - Chi so: Macro-F1 (nguong 0.5), PR-AUC, ROC-AUC, Precision/Recall phishing, xac suat TB moi lop.
    PR-AUC phai doc cung ty le phishing cua D2 (PR-AUC cua mo hinh doan ngau nhien = ty le do).

Luu y khi dien giai:
  - D2 la du lieu 2015-2019, khac thoi ky voi du lieu 2026; hai lop D2 khac nhau rat xa ve gia tri giao dich;
    D2 khong co gas (dat gia tri gia) nen CHI danh gia mo hinh khong dung thong tin gas cua canh
    (GraphSAGE, GATv2, PEAE-style, advtrain_peae); KHONG danh gia ra_peae.
  - Hieu nang thap co the do khac biet du lieu, khong han do mo hinh kem.

Chay:
  python stage8_eval_d2.py --max_targets 100       # thu nhanh
  python stage8_eval_d2.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import stage9_k2_upgrade as k2
import stage6_adversarial_attacks as s6
from stage5_train_baselines import MODELS

# ten phuong phap -> (kien truc trong MODELS, mau ten file checkpoint, nguon: stage5 | defense)
METHODS = {
    "graphsage":     ("graphsage", "model_graphsage_seed{s}.pt", "stage5"),
    "gatv2":         ("gatv2",     "model_gatv2_seed{s}.pt",     "stage5"),
    "peaegnn":       ("peaegnn",   "model_peaegnn_seed{s}.pt",   "stage5"),
    "advtrain_peae": ("peaegnn",   "model_advtrain_peae_seed{s}.pt", "defense"),
}


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    prec, rec, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "prec_phishing": float(prec[1]), "rec_phishing": float(rec[1]), "rec_normal": float(rec[0]),
            "mean_p_phishing_class": float(p[y == 1].mean()), "mean_p_normal_class": float(p[y == 0].mean()),
            "pred_phishing_rate": float(pred.mean()), "n": int(len(y)), "phishing_rate": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2base", default=r"D:\NCKH\Final\D2base")
    ap.add_argument("--stage5_dir", default=r"D:\NCKH\Final\results\stage5_k2")
    ap.add_argument("--defense_dir", default=r"D:\NCKH\Final\results\stage7_k2")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--max_targets", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = Path(a.d2base)
    s6._set_base(base)
    stage5_dir, defense_dir = Path(a.stage5_dir), Path(a.defense_dir)
    out = Path(a.out) if a.out else base
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = torch.load(stage5_dir / "feat_stats.pt", map_location="cpu", weights_only=True)
    in_dim = int(stats["node_mean"].numel())
    models = {}
    for m in a.methods:
        arch, pat, src = METHODS[m]
        for s in a.seeds:
            ck = (stage5_dir if src == "stage5" else defense_dir) / pat.format(s=s)
            if not ck.exists():
                print(f"  [thieu] {ck}")
                continue
            net = MODELS[arch](in_dim=in_dim).to(device)
            net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            net.eval()
            models[(m, s)] = net
    if not models:
        sys.exit("Khong nap duoc mo hinh nao.")
    print(f"Da nap {len(models)} mo hinh: " + ", ".join(f"{m}/s{s}" for m, s in models))

    split = pd.read_csv(k2.SPLIT_CSV)
    t_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600
    label_of = dict(zip(split["address"].str.lower(), split["label"].astype(int)))
    rows = split.copy()
    if a.max_targets:
        rows = rows.sample(n=min(a.max_targets, len(rows)), random_state=0)
    rows = rows.reset_index(drop=True)
    print(f"D2: {len(rows)} target (phishing={int((rows['label'] == 1).sum())}) | cutoff = lich su day du")

    y_all, part_all, probs = [], [], {k: [] for k in models}
    n_none, t0 = 0, time.time()
    for i0 in tqdm(range(0, len(rows), a.chunk), unit="chunk"):
        graphs, ys, ps = [], [], []
        for _, r in rows.iloc[i0:i0 + a.chunk].iterrows():
            addr, lab = r["address"].lower(), int(r["label"])
            g = s6.build_graph(s6.TargetCache(addr, lab, t_end, label_of))
            if g is None:
                n_none += 1
                continue
            graphs.append(g); ys.append(lab); ps.append(r["partition"])
        if not graphs:
            continue
        y_all += ys; part_all += ps
        for k, net in models.items():
            probs[k] += list(s6.predict_graphs(net, graphs, stats, device))
    print(f"\nThoi gian: {(time.time() - t0) / 60:.1f} phut | khong dung duoc graph: {n_none}")

    y_all, part_all = np.array(y_all), np.array(part_all)
    ref = {}
    rj = stage5_dir / "results_all.json"
    if rj.exists():
        ref = {k: v["agg"] for k, v in json.loads(rj.read_text(encoding="utf-8")).items()}

    results = {"n_targets": int(len(y_all)), "phishing_rate": float(y_all.mean()), "per_seed": {}, "agg": {}}
    for m in a.methods:
        ks = [k for k in models if k[0] == m]
        if not ks:
            continue
        results["per_seed"][m] = {f"seed{s}": {"all": metrics(y_all, probs[(m, s)]),
                                               **({"test_partition": metrics(y_all[part_all == "test"], np.array(probs[(m, s)])[part_all == "test"])}
                                                  if (part_all == "test").any() else {})}
                                  for _, s in ks}
        results["agg"][m] = {key: (float(np.mean([results["per_seed"][m][f"seed{s}"]["all"][key] for _, s in ks])),
                                   float(np.std([results["per_seed"][m][f"seed{s}"]["all"][key] for _, s in ks])))
                             for key in ("macro_f1", "pr_auc", "roc_auc", "rec_phishing", "prec_phishing",
                                         "mean_p_phishing_class", "mean_p_normal_class", "pred_phishing_rate")}
    (out / "stage8_zeroshot_d2.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    pr = results["phishing_rate"]
    print(f"\n{'=' * 110}\n D2 zero-shot (toan bo {len(y_all)} target; ty le phishing = {100 * pr:.1f}% -> PR-AUC ngau nhien ~ {pr:.2f})\n{'=' * 110}")
    print(f"{'Mo hinh':<15s}{'MacroF1':>16s}{'PR-AUC':>16s}{'ROC-AUC':>16s}{'Recall phish':>16s}{'P(phish)|phish':>16s}{'P(phish)|normal':>17s}")
    for m, ag in results["agg"].items():
        f = lambda k: f"{ag[k][0]:.3f}±{ag[k][1]:.3f}"
        print(f"{m:<15s}{f('macro_f1'):>16s}{f('pr_auc'):>16s}{f('roc_auc'):>16s}{f('rec_phishing'):>16s}"
              f"{f('mean_p_phishing_class'):>16s}{f('mean_p_normal_class'):>17s}")
    if ref:
        print("\nDoi chieu: test 2026 (Stage 5) macro-F1 / PR-AUC:  " + " | ".join(
            f"{k} {v['macro_f1']['mean']:.3f} / {v['pr_auc']['mean']:.3f}" for k, v in ref.items()))
    print(f"\nSaved -> {out / 'stage8_zeroshot_d2.json'}")


if __name__ == "__main__":
    main()
