"""
Kiem dinh thong ke + phan tich ty le phishing thap cho Stage 7, tu cac file eval_probs_seed*.npz da luu.
(Khong chay lai mo hinh. Thay cho Stage 4 vi khong co du lieu normal bo sung.)

PHAN 1 - Kiem dinh GHEP CAP (paired bootstrap tren cac target test, giu nguyen cap cua cung mot target):
  Voi moi seed, so sanh hai mo hinh A va B (mac dinh: moi phong thu vs baseline; ra_peae vs advtrain_peae):
    - Delta recall tai cung FPR khi bi tan cong (nguong chon tren vi normal SACH, tinh lai trong tung lan bootstrap)
    - Delta ROC-AUC sach (kiem tra phong thu co lam mat hieu nang sach khong)
  Bao cao khoang tin cay 95% (percentile) va so seed ma khoang tin cay nam han o phia duong.
  LUU Y: bootstrap chi the hien do bat dinh do CHON MAU target, KHONG gom do bien thien giua cac seed huan luyen.
  Voi chi 3 seed khong the kiem dinh co y nghia theo seed; dung ket hop: dau dong nhat qua seed + CI trong tung seed.

PHAN 2 - Ty le phishing thap (phishing : normal = 1:k): lay mau con (KHONG lap) cac vi phishing, giu NGUYEN so normal, lap --reps lan.
    - PR-AUC sach va khi bi tan cong, F1(phishing) o nguong 0.5
    - Precision tai cung FPR khi bi tan cong (tinh tu recall@FPR va ty le phishing).
  Ty le thuc cua tap test la ~ 594 : 814; 1:5 va 1:10 tao bang cach giam so phishing.

Chay:
  python stats_stage7.py                                   # ADAPTIVE-STC-LV, b=20 100
  python stats_stage7.py --tag hv --attack ADAPTIVE-STC
  python stats_stage7.py --boot 2000 --reps 500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage7_defense_k2")


def arrays(z, m, atk, b):
    y = z["labels"].astype(int)
    pc = z[f"{m}__clean"]
    pe = pc.copy()
    pe[y == 1] = z[f"{m}__{atk}__{b}"]
    return y, pc, pe


def rec_at_fpr(y, pc, pe, fpr):
    thr = np.quantile(pc[y == 0], 1 - fpr)
    return float((pe[y == 1] > thr).mean())


def boot_delta(za, zb, ma, mb, atk, b, fpr, B, rng):
    """Delta (A - B) cua recall@FPR khi bi tan cong va cua ROC-AUC sach; bootstrap ghep cap."""
    y, pca, pea = arrays(za, ma, atk, b)
    _, pcb, peb = arrays(zb, mb, atk, b)
    n = len(y)
    d_rec, d_auc = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        yi = y[i]
        d_rec.append(rec_at_fpr(yi, pca[i], pea[i], fpr) - rec_at_fpr(yi, pcb[i], peb[i], fpr))
        d_auc.append(roc_auc_score(yi, pca[i]) - roc_auc_score(yi, pcb[i]))
    point_rec = rec_at_fpr(y, pca, pea, fpr) - rec_at_fpr(y, pcb, peb, fpr)
    point_auc = roc_auc_score(y, pca) - roc_auc_score(y, pcb)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return point_rec, ci(d_rec), point_auc, ci(d_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--attack", default="ADAPTIVE-STC-LV")
    ap.add_argument("--budgets", nargs="+", type=int, default=[20, 100])
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--ratios", nargs="+", type=int, default=[5, 10], help="1:k")
    ap.add_argument("--baseline", default="baseline_peae")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d, sfx = Path(a.dir), (f"_{a.tag}" if a.tag else "")
    rng = np.random.default_rng(a.seed)

    Z = {}
    for s in a.seeds:
        f = d / f"eval_probs_seed{s}{sfx}.npz"
        if f.exists():
            Z[s] = np.load(f)
        else:
            print(f"  [thieu] {f.name}")
    if not Z:
        sys.exit("Khong co file eval_probs nao.")
    seeds = sorted(Z)
    methods = sorted({k.split("__")[0] for k in Z[seeds[0]].keys() if k != "labels"},
                     key=lambda m: (m != a.baseline, m))
    md = []

    def emit(x=""):
        print(x)
        md.append(x)

    pairs = [(m, a.baseline) for m in methods if m != a.baseline]
    if "ra_peae" in methods and "advtrain_peae" in methods:
        pairs.append(("ra_peae", "advtrain_peae"))

    # ---------------- PHAN 1 ----------------
    emit(f"## PHAN 1 - Kiem dinh ghep cap: {a.attack}, recall tai FPR={int(100 * a.fpr)}% khi bi tan cong, {a.boot} lan bootstrap")
    emit("Delta = (mo hinh A) - (mo hinh B), don vi diem %; CI 95% theo bootstrap tren target test (ghep cap). "
         "'+' = co khoang tin cay nam han o phia duong.\n")
    for b in a.budgets:
        emit(f"\n**Ngan sach b={b}**\n")
        emit("| A so voi B | " + " | ".join(f"seed {s}" for s in seeds) + " | so seed A hon B (CI>0) |")
        emit("|---|" + "---|" * (len(seeds) + 1))
        for ma, mb in pairs:
            cells, wins = [], 0
            for s in seeds:
                pr, ci, _, _ = boot_delta(Z[s], Z[s], ma, mb, a.attack, b, a.fpr, a.boot, rng)
                sig = ci[0] > 0
                wins += sig
                cells.append(f"{100 * pr:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]{' +' if sig else ''}")
            emit(f"| {ma} vs {mb} | " + " | ".join(cells) + f" | {wins}/{len(seeds)} |")

    emit(f"\n\n**Hieu nang SACH: delta ROC-AUC (A - B)**, CI 95% ghep cap (khong am han = khong mat hieu nang sach)\n")
    emit("| A so voi B | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    emit("|---|" + "---|" * len(seeds))
    for ma, mb in pairs:
        cells = []
        for s in seeds:
            _, _, pa, ci = boot_delta(Z[s], Z[s], ma, mb, a.attack, a.budgets[0], a.fpr, a.boot, rng)
            cells.append(f"{pa:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        emit(f"| {ma} vs {mb} | " + " | ".join(cells) + " |")

    # ---------------- PHAN 2 ----------------
    emit(f"\n\n## PHAN 2 - Ty le phishing thap ({a.reps} lan lay mau lai; trung binh ± do lech chuan qua {len(seeds)} seed)")
    emit("Ty le thuc cua tap test ~ phishing : normal = "
         f"{int((Z[seeds[0]]['labels'] == 1).sum())} : {int((Z[seeds[0]]['labels'] == 0).sum())}. "
         f"Tan cong: {a.attack}, b={a.budgets[-1]}. Precision@FPR: nguong chon tren vi normal sach (FPR={int(100 * a.fpr)}%).\n")
    b = a.budgets[-1]
    ratios = [0] + a.ratios   # 0 = ty le thuc
    emit("| Mo hinh | Ty le | PR-AUC sach | PR-AUC khi tan cong | F1 phishing@0.5 sach | Precision@FPR sach | Precision@FPR khi tan cong |")
    emit("|---|---|---|---|---|---|---|")
    for m in methods:
        for k in ratios:
            vals = {key: [] for key in ("pr_c", "pr_a", "f1", "pre_c", "pre_a")}
            for s in seeds:
                y, pc, pe = arrays(Z[s], m, a.attack, b)
                ph_idx, nm_idx = np.where(y == 1)[0], np.where(y == 0)[0]
                n_ph = len(ph_idx) if k == 0 else max(5, len(nm_idx) // k)
                reps = 1 if k == 0 else a.reps
                acc = {key: [] for key in vals}
                for _ in range(reps):
                    sel = ph_idx if k == 0 else rng.choice(ph_idx, size=min(n_ph, len(ph_idx)), replace=False)
                    idx = np.concatenate([sel, nm_idx])
                    yy = y[idx]
                    acc["pr_c"].append(average_precision_score(yy, pc[idx]))
                    pa = pc[idx].copy(); pa[: len(sel)] = pe[sel]
                    acc["pr_a"].append(average_precision_score(yy, pa))
                    acc["f1"].append(f1_score(yy, (pc[idx] >= 0.5).astype(int), pos_label=1, zero_division=0))
                    thr = np.quantile(pc[nm_idx], 1 - a.fpr)
                    pi = len(sel) / len(idx)
                    for tag, p_ph in (("c", pc[sel]), ("a", pe[sel])):
                        r = float((p_ph > thr).mean())
                        acc[f"pre_{tag}"].append(r * pi / (r * pi + a.fpr * (1 - pi)) if r > 0 else 0.0)
                for key in vals:
                    vals[key].append(float(np.mean(acc[key])))
            f = lambda key: f"{np.mean(vals[key]):.3f} ± {np.std(vals[key]):.3f}"
            emit(f"| {m} | {'thuc' if k == 0 else f'1:{k}'} | {f('pr_c')} | {f('pr_a')} | {f('f1')} | {f('pre_c')} | {f('pre_a')} |")

    out = d / f"stage7_stats{sfx}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
