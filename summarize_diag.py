"""
Gop ket qua diag_feature_sensitivity.py (diag_feature_<mo_hinh>_seed<seed>.json) thanh bang so sanh cac mo hinh.

Bang A: thay dac trung cua CHINH vi bang gia tri dien hinh cua normal -> % vi phishing bi lat (tat ca / tung nhom).
Bang B: thay dac trung hang xom bang MAU THAT (boc tu graph NORMAL) so voi DOI CHUNG (boc tu graph PHISHING).
        'Hieu' = (% lat voi mau normal) - (% lat voi mau phishing, doi chung): chi phan nay moi la bang chung mo hinh
        doc 'do giong normal' cua hang xom; phan con lai la do chinh viec thay the gay nhieu.

Cac dong thay bang MOT VECTOR HANG SO ('hop-1 -> normal', ...) bi bo: gia tri hang so la ngoai phan phoi, lat ~100% o
mot so mo hinh la hien tuong nhan tao, khong doc duoc.

Chay:
  python summarize_diag.py                          # seed 42
  python summarize_diag.py --seeds 42 43 44
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage6_attacks_k2")
MODELS = ["peaegnnv2", "graphsage", "gatv2"]

A_ROWS = [
    ("Tat ca dac trung cua vi", "target -> normal (tat ca dac trung)"),
    ("  so giao dich vao/ra (counts)", "target -> normal: counts (in/out count)"),
    ("  so doi tac khac nhau (unique peers)", "target -> normal: unique peers"),
    ("  tong gia tri (value sums)", "target -> normal: value sums"),
    ("  gia tri trung binh (value means)", "target -> normal: value means"),
    ("  thoi gian hoat dong (active_span)", "target -> normal: active_span"),
    ("[doi chung] thay bang gia tri dien hinh cua PHISHING", "[DOI CHUNG] target -> phishing dien hinh"),
]
B_ROWS = [
    ("Hang xom bac 1", "hop-1 -> mau thuc tu NORMAL", "[DOI CHUNG] hop-1 -> mau thuc tu PHISHING"),
    ("Hang xom bac 2", "hop-2 -> mau thuc tu NORMAL", "[DOI CHUNG] hop-2 -> mau thuc tu PHISHING"),
    ("Bac 1 + bac 2", "hop-1+hop-2 -> mau thuc tu NORMAL", "[DOI CHUNG] hop-1+hop-2 -> mau thuc tu PHISHING"),
]


def fmt(v, pct=True, sign=False):
    v = np.array(v, float)
    f = (lambda x: f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f(v.mean()) + (f" ± {100 * v.std():.1f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DIR))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    a = ap.parse_args()
    d = Path(a.dir)

    data = {}
    for m in a.models:
        for s in a.seeds:
            f = d / f"diag_feature_{m}_seed{s}.json"
            if f.exists():
                data.setdefault(m, []).append(json.loads(f.read_text(encoding="utf-8")))
            else:
                print(f"  [thieu] {f.name}")
    models = [m for m in a.models if m in data]
    if not models:
        sys.exit("Khong co file nao.")
    md = []

    def emit(x=""):
        print(x)
        md.append(x)

    emit(f"## Do nhay dac trung (RQ2) - seed {a.seeds}; so vi phishing mo hinh bat dung luc sach: " +
         ", ".join(f"{m} {int(np.mean([r['n_correct_clean'] for r in data[m]]))}" for m in models))
    emit("\n**Bang A. Thay dac trung cua CHINH vi bang gia tri dien hinh cua normal: % vi phishing bi lat** (khong phai giao dich that)\n")
    emit("| Thay phan nao | " + " | ".join(models) + " |")
    emit("|---|" + "---|" * len(models))
    for label, key in A_ROWS:
        emit(f"| {label} | " + " | ".join(
            fmt([r["interventions"][key]["flip_rate"] for r in data[m]]) for m in models) + " |")

    emit("\n**Bang B. Thay hang xom bang MAU THAT: % vi phishing bi lat**\n")
    emit("Moi o: mau tu NORMAL / doi chung (mau tu PHISHING) / HIEU = phan dac hieu cua lop (diem %)\n")
    emit("| Thay phan nao | " + " | ".join(models) + " |")
    emit("|---|" + "---|" * len(models))
    for label, kn, kp in B_ROWS:
        cells = []
        for m in models:
            n = [r["interventions"][kn]["flip_rate"] for r in data[m]]
            p = [r["interventions"][kp]["flip_rate"] for r in data[m]]
            diff = [x - y for x, y in zip(n, p)]
            cells.append(f"{fmt(n)} / {fmt(p)} / **{fmt(diff, sign=True)}**")
        emit(f"| {label} | " + " | ".join(cells) + " |")
    emit("\nLuu y: 'HIEU' gan 0 hoac am = viec thay the gay nhieu chu khong phai mo hinh doc 'do giong normal'. "
         "Tat ca la thay gia tri truc tiep (khong qua giao dich); thay TOAN BO hang xom la can thiep manh hon nhieu so voi "
         "nhung gi tan cong co the lam (them vai nut hoac vai tram nut).")
    out = d / "diag_summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
