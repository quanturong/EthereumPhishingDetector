"""
Tong hop ket qua Stage 6: trung binh +- do lech chuan qua cac seed (3 mo hinh x 3 seed).
Chay:   python summarize_stage6.py                                  # 6 tan cong goc (file khong co tag)
        python summarize_stage6.py --tag adaptive_lv                 # cac file results_*_adaptive_lv.json
        python summarize_stage6.py --dir D:\\NCKH\\Final\\ethphishgraph2026_stage6_attacks_k2
Ket qua: in bang ra man hinh + luu stage6_summary.csv va stage6_summary.md trong cung thu muc.

Luu y khi viet khoa luan:
  - Do lech chuan o day (np.std, ddof=0, giong Stage 5) chi phan anh viec KHOI TAO MO HINH khac
    nhau. Ca 3 seed dung cung tap test, cung cach chia va cung cac giao dich gia (attack_seed=42).
  - ASR duoc tinh tren cac mau phishing ma TUNG mo hinh phat hien dung luc clean, nen mau so
    khac nhau giua cac seed/mo hinh.
  - ASR rat thap (vai %) thi 1 mau lat ~ 0.2%; chenh lech nho giua cac tan cong co the chi la nhieu.
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

DEFAULT_DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage6_attacks_k2")
MODELS = ["peaegnn", "peaegnnv2", "graphsage", "gatv2"]
ATTACKS_ALL = ["TOH", "TIH", "BPH-T2P", "BPH-P2T", "STC", "MWR", "ADAPTIVE-STC", "STC-HV",
               "ADAPTIVE-STC-LV", "COMBINED", "GREEDY-QUERY"]
KNOWN = ["TOH", "TIH", "BPH-T2P", "BPH-P2T"]      # moi tan cong khac = unseen (khong dung de huan luyen)


def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


def fmt(m, s, pct=True, nd=1):
    if np.isnan(m):
        return "-"
    return f"{m * 100:.{nd}f} ± {s * 100:.{nd}f}" if pct else f"{m:.3f} ± {s:.3f}"


def load(dirpath: Path, model: str, seeds, tag=""):
    out = []
    for s in seeds:
        f = dirpath / f"results_{model}_seed{s}{('_' + tag) if tag else ''}.json"
        if not f.exists():
            print(f"  [thieu] {f.name}")
            continue
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tag", default="", help="hau to cua file ket qua Stage 6 (vd adaptive_lv)")
    ap.add_argument("--models", nargs="+", default=None, help="chi tong hop cac mo hinh nay (vd peaegnnv2 graphsage gatv2); mac dinh: tat ca")
    ap.add_argument("--attacks", nargs="+", default=None, help="mac dinh: moi tan cong co trong file")
    args = ap.parse_args()
    d = Path(args.dir)

    runs = {m: load(d, m, args.seeds, args.tag) for m in (args.models or MODELS)}
    runs = {m: r for m, r in runs.items() if r}
    if not runs:
        raise SystemExit(f"Khong tim thay file results_*_seed*.json trong {d}")
    first = next(iter(runs.values()))[0]
    ATTACKS = args.attacks or [a for a in ATTACKS_ALL if a in first["attacks"]]
    budgets = [b["budget"] for b in first["attacks"][ATTACKS[0]]["budgets"]]
    sfx = f"_{args.tag}" if args.tag else ""

    csv_rows, md = [], []

    def emit(line=""):
        print(line)
        md.append(line)

    emit("## Hieu nang sach (clean) tren graph dung lai - trung binh ± do lech chuan qua seed")
    emit(f"| Mo hinh | So seed | Macro-F1 | Phishing recall | So mau phishing dung (TB) |")
    emit("|---|---|---|---|---|")
    for m, rs in runs.items():
        f1 = ms([r["clean_macro_f1"] for r in rs]); rc = ms([r["clean_phish_recall"] for r in rs])
        nc = np.mean([r["n_correct_clean"] for r in rs])
        emit(f"| {m} | {len(rs)} | {fmt(*f1, nd=2)} | {fmt(*rc, nd=1)} % | {nc:.0f}/{rs[0]['n_phishing']} |")
        csv_rows.append([m, "clean", "", "macro_f1", *f1, len(rs)])
        csv_rows.append([m, "clean", "", "phish_recall", *rc, len(rs)])

    for m, rs in runs.items():
        emit(f"\n## ASR (%) - {m} - trung binh ± do lech chuan qua {len(rs)} seed")
        emit("| Tan cong | nhom | " + " | ".join(f"b={b}" for b in budgets) + " | TB cac budget |")
        emit("|---|---|" + "---|" * (len(budgets) + 1))
        for a in ATTACKS:
            cells, means = [], []
            for i, b in enumerate(budgets):
                vals = [r["attacks"][a]["budgets"][i]["asr"] for r in rs]
                mm, ss = ms(vals)
                cells.append(fmt(mm, ss)); means.append([v for v in vals])
                csv_rows.append([m, a, b, "asr", mm, ss, len(rs)])
            avg = ms([np.mean([means[i][k] for i in range(len(budgets))]) for k in range(len(rs))])
            csv_rows.append([m, a, "mean", "asr", *avg, len(rs)])
            grp = "known" if a in KNOWN else "unseen"
            emit(f"| {a} | {grp} | " + " | ".join(cells) + f" | {fmt(*avg)} |")
        anyv = ms([r["asr_any"] for r in rs])
        csv_rows.append([m, "ANY", "", "asr_any", *anyv, len(rs)])
        emit(f"\nASR_any (lat boi it nhat 1 tan cong/ngan sach): {fmt(*anyv)} %  "
             f"(so mau lat TB: {np.mean([r['n_flipped_any'] for r in rs]):.1f})")

    emit("\n## Huong tac dong len P(phishing) - budget lon nhat (thay doi so voi clean, don vi diem xac suat)")
    emit("| Mo hinh | " + " | ".join(ATTACKS) + " |")
    emit("|---|" + "---|" * len(ATTACKS))
    for m, rs in runs.items():
        cells = []
        for a in ATTACKS:
            dps = []
            for r in rs:
                p = r["attacks"][a]["budgets"][-1].get("mean_p_phish_attacked")
                c = r.get("mean_p_phish_clean")
                if p is not None and c is not None:
                    dps.append(p - c)
            mm, ss = ms(dps)
            cells.append("-" if np.isnan(mm) else f"{mm:+.3f} ± {ss:.3f}")
        emit(f"| {m} | " + " | ".join(cells) + " |")
    emit("\n(am = tan cong lam mo hinh bot tin la phishing; duong = tan cong vo tinh giup mo hinh nhan ra phishing hon)")

    if all("mean_value_eth" in r["attacks"][ATTACKS[0]]["budgets"][-1] for rs in runs.values() for r in rs):
        emit(f"\n## Chi phi trung binh / mau o ngan sach lon nhat (b={budgets[-1]}) - qua cac seed")
        emit("| Mo hinh | Tan cong | So tx | Tong value (ETH) | Tong gas (ETH) |")
        emit("|---|---|---|---|---|")
        for m, rs in runs.items():
            for a in ATTACKS:
                x = [r["attacks"][a]["budgets"][-1] for r in rs]
                emit(f"| {m} | {a} | {np.mean([i['amt'] for i in x]):.1f} | "
                     f"{np.mean([i['mean_value_eth'] for i in x]):.3f} | {np.mean([i['mean_gas_eth'] for i in x]):.4f} |")

    emit("\n## Kiem tra du lieu")
    for m, rs in runs.items():
        inv = sum(r["n_invalid_tx"] for r in rs); sk = sum(r["n_skipped"] for r in rs)
        emit(f"- {m}: {len(rs)} seed | giao dich khong hop le: {inv} | target bi bo qua: {sk}")

    import shutil, time                       # luon sao luu file cu truoc khi ghi de (tranh mat bang tom tat day du)
    for ext in ("csv", "md"):
        old_f = d / f"stage6_summary{sfx}.{ext}"
        if old_f.exists():
            bak = d / f"stage6_summary{sfx}.{ext}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(old_f, bak)
            print(f"[sao luu] {old_f.name} -> {bak.name}")
    with open(d / f"stage6_summary{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "attack", "budget", "metric", "mean", "std", "n_seeds"])
        w.writerows(csv_rows)
    (d / f"stage6_summary{sfx}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved -> {d / f'stage6_summary{sfx}.csv'}\nSaved -> {d / f'stage6_summary{sfx}.md'}")


if __name__ == "__main__":
    main()
