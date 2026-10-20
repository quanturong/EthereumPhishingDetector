"""
Tong hop ket qua danh gia Stage 7 qua cac seed (file eval_seed42.json, eval_seed43.json, ...).
Chay:  python summarize_stage7.py                      # file eval_seed{42,43,44}.json
       python summarize_stage7.py --tag big            # file eval_seed{S}_big.json
Luu y: so sanh voi mo hinh goc la GHEP THEO SEED (cung seed, cung giao dich gia). Do lech chuan
chi phan anh khoi tao mo hinh khac nhau. ASR thap (vai %) thi chenh lech nho la nhieu.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DEFAULT_DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage7_defense_k2")
def ms(v):
    v = np.array([x for x in v if x is not None], float)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"),) * 2
def pct(m, s): return "-" if np.isnan(m) else f"{100*m:.1f} ± {100*s:.1f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR)); ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--baseline", default="baseline_peae")
    a = ap.parse_args(); d = Path(a.dir); sfx = f"_{a.tag}" if a.tag else ""
    R = {}
    for s in a.seeds:
        f = d / f"eval_seed{s}{sfx}.json"
        if f.exists(): R[s] = json.loads(f.read_text(encoding="utf-8"))
        else: print(f"  [thieu] {f.name}")
    if not R: raise SystemExit("Khong co file eval nao.")
    seeds = sorted(R); methods = list(R[seeds[0]]); base = a.baseline
    if base not in methods: raise SystemExit(f"Khong co mo hinh goc {base}")
    attacks = list(R[seeds[0]][base]["attacks"])
    budgets = [b["budget"] for b in R[seeds[0]][base]["attacks"][attacks[0]]["budgets"]]
    md, rows = [], []
    def emit(x=""): print(x); md.append(x)
    def get(m, key): return [R[s][m][key] for s in seeds]
    def diff(m, f):   # ghep theo seed: (mo hinh - goc)
        return [f(R[s][m]) - f(R[s][base]) for s in seeds
                if f(R[s][m]) is not None and f(R[s][base]) is not None]

    emit(f"## Hieu nang sach (trung binh ± do lech chuan qua {len(seeds)} seed: {seeds}); delta ghep theo seed so voi {base}")
    emit("| Mo hinh | Macro-F1 (%) | Recall phishing (%) | dMacro-F1 (diem %) |"); emit("|---|---|---|---|")
    for m in methods:
        f1, rc = ms(get(m, "clean_macro_f1")), ms(get(m, "clean_phish_recall"))
        dd = ms(diff(m, lambda r: r["clean_macro_f1"]))
        emit(f"| {m} | {pct(*f1)} | {pct(*rc)} | {'-' if m == base else f'{100*dd[0]:+.1f} ± {100*dd[1]:.1f}'} |")
        rows += [[m, "clean", "", "macro_f1", *f1, len(seeds)], [m, "clean", "", "recall", *rc, len(seeds)]]

    emit(f"\n## ASR (%) trung binh cac ngan sach {budgets}, theo tan cong")
    emit("| Mo hinh | " + " | ".join(attacks) + " | known | unseen | ANY |"); emit("|---|" + "---|" * (len(attacks) + 3))
    for m in methods:
        cells = [pct(*ms([R[s][m]["attacks"][x]["mean_asr"] for s in seeds])) for x in attacks]
        cells += [pct(*ms(get(m, "mean_asr_known"))), pct(*ms(get(m, "mean_asr_unseen"))), pct(*ms(get(m, "asr_any")))]
        emit(f"| {m} | " + " | ".join(cells) + " |")

    emit("\n## ASR (%) theo ngan sach, tung tan cong")
    for x in attacks:
        emit(f"\n**{x}**\n\n| Mo hinh | " + " | ".join(f"b={b}" for b in budgets) + " |"); emit("|---|" + "---|" * len(budgets))
        for m in methods:
            cells = []
            for i, b in enumerate(budgets):
                mm, ss = ms([R[s][m]["attacks"][x]["budgets"][i]["asr"] for s in seeds]); cells.append(pct(mm, ss))
                rows.append([m, x, b, "asr", mm, ss, len(seeds)])
            emit(f"| {m} | " + " | ".join(cells) + " |")

    unseen = [x for x in attacks if x not in ("TOH", "TIH", "BPH-T2P", "BPH-P2T")]
    emit("\n## Robust recall (%) = ty le TOAN BO vi phishing van bi phat hien sau tan cong (khong dieu kien hoa)")
    emit("(cong bang hon ASR khi cac mo hinh co recall sach khac nhau; so voi recall sach o bang dau)")
    for x in unseen:
        emit(f"\n**{x}**\n\n| Mo hinh | recall sach | " + " | ".join(f"b={b}" for b in budgets) + " |"); emit("|---|---|" + "---|" * len(budgets))
        for m in methods:
            cells = [pct(*ms([R[s][m]["attacks"][x]["budgets"][i]["robust_recall"] for s in seeds])) for i in range(len(budgets))]
            emit(f"| {m} | {pct(*ms(get(m, 'clean_phish_recall')))} | " + " | ".join(cells) + " |")

    emit(f"\n## Thay doi ASR so voi {base} (diem %, ghep theo seed; am = phong thu giam ASR)")
    emit("| Mo hinh | known | unseen | ANY | so seed ma ANY giam |"); emit("|---|---|---|---|---|")
    for m in methods:
        if m == base: continue
        k, u, an = (ms(diff(m, lambda r, key=key: r[key])) for key in ("mean_asr_known", "mean_asr_unseen", "asr_any"))
        dany = diff(m, lambda r: r["asr_any"])
        sg = lambda t: "-" if np.isnan(t[0]) else f"{100*t[0]:+.1f} ± {100*t[1]:.1f}"
        emit(f"| {m} | {sg(k)} | {sg(u)} | {sg(an)} | {sum(x < 0 for x in dany)}/{len(dany)} |")

    emit("\n## Tung seed (de thay do bien thien): ASR_any (%) va macro-F1 sach (%)")
    emit("| Mo hinh | " + " | ".join(f"seed {s}" for s in seeds) + " |"); emit("|---|" + "---|" * len(seeds))
    for m in methods:
        emit(f"| {m} | " + " | ".join(f"{100*R[s][m]['asr_any']:.1f} / {100*R[s][m]['clean_macro_f1']:.1f}" for s in seeds) + " |")
    with open(d / f"stage7_summary{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "attack", "budget", "metric", "mean", "std", "n_seeds"]); w.writerows(rows)
    (d / f"stage7_summary{sfx}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved -> {d / f'stage7_summary{sfx}.md'}")
if __name__ == "__main__": main()
