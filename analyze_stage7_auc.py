"""
Phan tich do ben KHONG PHU THUOC NGUONG cho Stage 7, tu cac file eval_probs_seed*.npz da luu.

Vi sao can: ASR / recall o nguong 0.5 bi anh huong boi viec mo hinh 'nghieng ve phishing' hay khong
(vd advtrain: mau tan cong deu gan nhan phishing -> mo hinh co the doan phishing nhieu hon -> kho bi lat).
Hai so do sau cong bang giua cac mo hinh co 'uu tien' khac nhau:
  - ROC-AUC / PR-AUC KHI BI TAN CONG: vi phishing da bi tan cong (xac suat sau tan cong) so voi vi NORMAL nguyen ven.
  - Recall tai CUNG ty le bao nham (FPR): nguong duoc chon tren cac vi normal SACH sao cho dung FPR% vi normal
    vuot nguong; sau do do ty le vi phishing DA BI TAN CONG van vuot nguong (khong dieu kien hoa).

Chay:
  python analyze_stage7_auc.py                    # file eval_probs_seed{42,43,44}.npz
  python analyze_stage7_auc.py --tag hv           # file eval_probs_seed{S}_hv.npz
  python analyze_stage7_auc.py --attack ADAPTIVE-STC-LV --fpr 0.05 0.1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage7_defense_k2")


def ms(v):
    v = np.array(v, float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def fmt(t, pct=False):
    m, s = t
    return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"


def recall_at_fpr(y, p_clean, p_eval, fpr):
    """Nguong chon tren vi normal SACH: dung fpr ty le normal vuot nguong. Do recall cua p_eval (vi phishing)."""
    thr = np.quantile(p_clean[y == 0], 1 - fpr)
    return float((p_eval[y == 1] > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--attack", default=None, help="mac dinh: moi tan cong khong thuoc nhom da biet")
    ap.add_argument("--fpr", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--baseline", default="baseline_peae")
    a = ap.parse_args()
    d = Path(a.dir)
    sfx = f"_{a.tag}" if a.tag else ""

    data = {}
    for s in a.seeds:
        f = d / f"eval_probs_seed{s}{sfx}.npz"
        if f.exists():
            data[s] = np.load(f)
        else:
            print(f"  [thieu] {f.name}")
    if not data:
        sys.exit("Khong co file eval_probs nao.")
    seeds = sorted(data)
    keys0 = list(data[seeds[0]].keys())
    methods = sorted({k.split("__")[0] for k in keys0 if k != "labels"}, key=lambda m: (m != a.baseline, m))
    atk_budget = sorted({(k.split("__")[1], int(k.split("__")[2])) for k in keys0 if k.count("__") == 2})
    known = {"TOH", "TIH", "BPH-T2P", "BPH-P2T"}
    attacks = [a.attack] if a.attack else sorted({x for x, _ in atk_budget if x not in known})
    md = []

    def emit(x=""):
        print(x)
        md.append(x)

    for atk in attacks:
        budgets = sorted(b for x, b in atk_budget if x == atk)
        emit(f"\n## {atk} - do ben khong phu thuoc nguong (trung binh ± do lech chuan qua {len(seeds)} seed: {seeds})")
        emit("(vi phishing sau tan cong vs vi normal NGUYEN VEN; recall tai cung FPR tinh tren TOAN BO vi phishing)\n")
        hdr = "| Mo hinh | " + " | ".join(
            ["AUC sach"] + [f"AUC b={b}" for b in budgets] + [f"PR-AUC b={budgets[-1]}"]
            + [f"recall@FPR{int(100 * f)}% sach" for f in a.fpr]
            + [f"recall@FPR{int(100 * f)}% b={b}" for f in a.fpr for b in budgets]) + " |"
        emit(hdr)
        emit("|---|" + "---|" * (hdr.count("|") - 2))
        per_method = {}
        for m in methods:
            auc_c, auc_b, pr_b, rec_c, rec_b = [], {b: [] for b in budgets}, [], {f: [] for f in a.fpr}, {(f, b): [] for f in a.fpr for b in budgets}
            for s in seeds:
                z = data[s]
                y = z["labels"].astype(int)
                pc = z[f"{m}__clean"]
                auc_c.append(roc_auc_score(y, pc))
                for f in a.fpr:
                    rec_c[f].append(recall_at_fpr(y, pc, pc, f))
                for b in budgets:
                    pe = pc.copy()
                    pe[y == 1] = z[f"{m}__{atk}__{b}"]
                    auc_b[b].append(roc_auc_score(y, pe))
                    if b == budgets[-1]:
                        pr_b.append(average_precision_score(y, pe))
                    for f in a.fpr:
                        rec_b[(f, b)].append(recall_at_fpr(y, pc, pe, f))
            per_method[m] = (auc_b, rec_b)
            cells = [fmt(ms(auc_c))] + [fmt(ms(auc_b[b])) for b in budgets] + [fmt(ms(pr_b))]
            cells += [fmt(ms(rec_c[f]), True) for f in a.fpr]
            cells += [fmt(ms(rec_b[(f, b)]), True) for f in a.fpr for b in budgets]
            emit(f"| {m} | " + " | ".join(cells) + " |")
        if a.baseline in per_method:
            emit(f"\nThay doi so voi {a.baseline} (ghep theo seed): AUC o ngan sach lon nhat b={budgets[-1]}, "
                 f"recall@FPR{int(100 * a.fpr[-1])}% o b={budgets[-1]} (diem %)")
            base_auc, base_rec = per_method[a.baseline]
            for m, (auc_b, rec_b) in per_method.items():
                if m == a.baseline:
                    continue
                dauc = np.array(auc_b[budgets[-1]]) - np.array(base_auc[budgets[-1]])
                drec = np.array(rec_b[(a.fpr[-1], budgets[-1])]) - np.array(base_rec[(a.fpr[-1], budgets[-1])])
                emit(f"  - {m}: dAUC = {dauc.mean():+.3f} ± {dauc.std():.3f} | drecall = {100 * drec.mean():+.1f} ± {100 * drec.std():.1f} diem "
                     f"| cai thien o {int((drec > 0).sum())}/{len(drec)} seed")
    out = d / f"stage7_auc_analysis{sfx}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
