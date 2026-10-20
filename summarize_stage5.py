"""
Tong hop Stage 5 cho cac mo hinh hien dung: peae_paper (mo hinh chinh), graphsage, gatv2.

Doc:
  - ket qua huan luyen:  results_all.json (graphsage, gatv2)  +  results_peae_paper.json
  - ROC-AUC:             1 hoac nhieu file stage5_rocauc.json (file sau ghi de file truoc theo tung mo hinh),
                         vd file cu (graphsage, gatv2, ...) + file moi chi co peae_paper.
Ghi ra thu muc --out:
  stage5_summary.md / stage5_summary.csv   bang tong hop (macro-F1, PR-AUC, ROC-AUC, precision, recall) trung binh ± do lech chuan
  stage5_rocauc_merged.json                ROC-AUC GOP chung 1 file cho cac mo hinh tren (khong sua file goc)

Chay (mac dinh dung duong dan cua ban):
  python summarize_stage5.py
  python summarize_stage5.py --roc D:\\NCKH\\Final\\results\\stage5_rocauc.json D:\\NCKH\\Final\\results\\roc_peae\\stage5_rocauc.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

R = Path(r"D:\NCKH\Final\results")


def ms(v, d=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_all", default=str(R / "stage5_k2" / "results_all.json"))
    ap.add_argument("--peae", default=str(R / "stage5_peae" / "results_peae_paper.json"))
    ap.add_argument("--roc", nargs="+", default=[str(R / "stage5_rocauc.json"), str(R / "roc_peae" / "stage5_rocauc.json")])
    ap.add_argument("--models", nargs="+", default=["peae_paper", "graphsage", "gatv2"])
    ap.add_argument("--out", default=str(R / "stage5_final"))
    a = ap.parse_args()

    res = {}
    for f in (a.results_all, a.peae):
        p = Path(f)
        if p.exists():
            res.update(json.loads(p.read_text(encoding="utf-8")))
        else:
            print(f"  [thieu] {p}")
    roc_ps = {}                    # model -> list theo seed
    for f in a.roc:
        p = Path(f)
        if not p.exists():
            print(f"  [thieu] {p}")
            continue
        for m, rows in json.loads(p.read_text(encoding="utf-8")).get("per_seed", {}).items():
            roc_ps[m] = rows       # file sau ghi de file truoc
    models = [m for m in a.models if m in res]
    if not models:
        sys.exit("Khong co ket qua nao.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, rows_csv = [], []
    emit = lambda x="": (print(x), md.append(x))

    emit("## Stage 5 - mo hinh hien dung (tap test 1.408 vi; trung binh ± do lech chuan qua seed)\n")
    emit("| Mo hinh | So seed | Macro-F1 | PR-AUC | **ROC-AUC** | Recall phishing | Precision phishing | F1 phishing |")
    emit("|---|---|---|---|---|---|---|---|")
    for m in models:
        ps = res[m]["per_seed"]
        g = lambda k: [s["test"][k] for s in ps]
        roc = [r["roc_auc"] for r in roc_ps.get(m, [])]
        emit(f"| {m} | {len(ps)} | {ms(g('macro_f1'))} | {ms(g('pr_auc'))} | {ms(roc) if roc else 'chua tinh'} | "
             f"{ms(g('rec_phishing'))} | {ms(g('prec_phishing'))} | {ms(g('f1_phishing'))} |")
        rows_csv.append({"model": m, "n_seed": len(ps), "macro_f1_mean": np.mean(g("macro_f1")), "macro_f1_std": np.std(g("macro_f1")),
                         "pr_auc_mean": np.mean(g("pr_auc")), "pr_auc_std": np.std(g("pr_auc")),
                         "roc_auc_mean": np.mean(roc) if roc else "", "roc_auc_std": np.std(roc) if roc else "",
                         "rec_phishing_mean": np.mean(g("rec_phishing")), "prec_phishing_mean": np.mean(g("prec_phishing"))})
    emit("\n### Theo tung seed (macro-F1 / ROC-AUC / epoch tot nhat)\n")
    emit("| Mo hinh | seed 1 | seed 2 | seed 3 |")
    emit("|---|---|---|---|")
    for m in models:
        cells = []
        for i, s in enumerate(res[m]["per_seed"]):
            r = roc_ps.get(m, [])
            roc = f"{r[i]['roc_auc']:.4f}" if i < len(r) else "-"
            cells.append(f"{s['test']['macro_f1']:.4f} / {roc} / ep {s['best_epoch']}")
        emit(f"| {m} | " + " | ".join(cells) + " |")

    # kiem tra nhat quan giua results va ROC (macro-F1 tinh lai phai trung khop)
    worst = 0.0
    for m in models:
        for s, r in zip(res[m]["per_seed"], roc_ps.get(m, [])):
            worst = max(worst, abs(s["test"]["macro_f1"] - r["macro_f1"]), abs(s["test"]["pr_auc"] - r["pr_auc"]))
    if roc_ps:
        emit(f"\nDoi chieu results vs ROC (macro-F1, PR-AUC tinh lai): lech lon nhat = {worst:.4f} -> " +
             ("KHOP" if worst <= 0.002 else "LECH: KHONG TIN ROC-AUC"))
    miss = [m for m in models if m not in roc_ps]
    if miss:
        emit(f"\nChua co ROC-AUC cho: {', '.join(miss)}")

    (out / "stage5_summary.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "stage5_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    (out / "stage5_rocauc_merged.json").write_text(
        json.dumps({"per_seed": {m: roc_ps[m] for m in models if m in roc_ps},
                    "agg": {m: {k: [float(np.mean([r[k] for r in roc_ps[m]])), float(np.std([r[k] for r in roc_ps[m]]))]
                                for k in ("roc_auc", "pr_auc", "macro_f1")} for m in models if m in roc_ps}}, indent=2),
        encoding="utf-8")
    print(f"\nSaved -> {out / 'stage5_summary.md'}\nSaved -> {out / 'stage5_summary.csv'}\nSaved -> {out / 'stage5_rocauc_merged.json'}")


if __name__ == "__main__":
    main()
