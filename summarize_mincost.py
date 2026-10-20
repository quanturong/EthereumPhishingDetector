"""
Gop ket qua MOC VUOT TOI THIEU (results_<mo_hinh>_seed<seed>_mincost_<nhom>.json) thanh bang so sanh theo SO GIAO DICH GIA.

Bang 1: so giao dich gia can de vuot 10% / 25% / 50% so vi phishing (vi bi 'dung ngay khi bi lua'); "chua dat" = khong dat
        duoc ty le do trong gioi han ngan sach da quet.
Bang 2: ty le vi phishing bi vuot khi chi dung toi da T giao dich (T = 30, 120, 300, 600, 1200); '-' neu tan cong do
        khong duoc quet toi T giao dich.
Luu y: 'vuot' = lan DAU TIEN mo hinh bi lat khi tang dan ngan sach; chi tinh tren vi phishing ma mo hinh bat DUNG luc sach;
       nhieu seed: trung binh ± do lech chuan giua seed; 1 seed: chi la gia tri don.

Chay:
  python summarize_mincost.py                          # seed 42
  python summarize_mincost.py --seeds 42 43 44
  python summarize_mincost.py --models peaegnnv2 graphsage gatv2 --tag_suffix ""
"""
import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage6_attacks_k2")
GROUPS = ["toh_tih", "bph", "mwr", "stc", "adaptive"]
UNITS = {"TOH": 1, "TIH": 1, "BPH-T2P": 2, "BPH-P2T": 2, "MWR": 3, "STC": 4, "ADAPTIVE-STC-LV": 6}
ORDER = ["TOH", "TIH", "BPH-T2P", "BPH-P2T", "MWR", "STC", "ADAPTIVE-STC-LV"]
CAPS = [30, 120, 300, 600, 1200]


def quantile_txs(min_txs, n, frac):
    need = ceil(frac * n)
    got = sorted(x for x in min_txs if x is not None)
    return float(got[need - 1]) if n and len(got) >= need else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DIR))
    ap.add_argument("--models", nargs="+", default=["peaegnnv2", "graphsage", "gatv2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--groups", nargs="+", default=GROUPS)
    ap.add_argument("--tag_suffix", default="")
    a = ap.parse_args()
    d = Path(a.dir)

    import re

    def load_results(m, s, tag):
        """Tra ve dict ket qua cho (mo hinh, seed, nhom): file chinh, hoac GOP DU cac manh *_shKofN. None neu thieu."""
        f = d / f"results_{m}_seed{s}_mincost_{tag}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        shards = sorted(d.glob(f"results_{m}_seed{s}_mincost_{tag}_sh*of*.json"))
        if not shards:
            print(f"  [thieu] {f.name}")
            return None
        nsh = int(re.search(r"_sh\d+of(\d+)\.json$", shards[0].name).group(1))
        got = sorted(int(re.search(r"_sh(\d+)of", p.name).group(1)) for p in shards)
        if got != list(range(1, nsh + 1)):
            print(f"  [thieu manh] {m} seed{s} {tag}: co {got}, can 1..{nsh}")
            return None
        parts = [json.loads(p.read_text(encoding="utf-8")) for p in shards]
        merged = {"n_eligible": sum(x["n_eligible"] for x in parts), "budgets": parts[0]["budgets"], "attacks": {}}
        for atk in parts[0]["attacks"]:
            merged["attacks"][atk] = {"min_txs": [t for x in parts for t in x["attacks"][atk]["min_txs"]]}
        print(f"  [gop {nsh} manh] {m} seed{s} {tag}: {merged['n_eligible']} vi du dieu kien")
        return merged

    # data[model][attack] = list theo seed cua dict(n, min_txs, cap)
    data = {}
    for m in a.models:
        for s in a.seeds:
            for g in a.groups:
                tag = g + (f"_{a.tag_suffix}" if a.tag_suffix else "")
                r = load_results(m, s, tag)
                if r is None:
                    continue
                for atk, v in r["attacks"].items():
                    cap = max(r["budgets"]) * UNITS[atk]
                    data.setdefault(m, {}).setdefault(atk, []).append(
                        {"n": r["n_eligible"], "min_txs": v["min_txs"], "cap": cap, "seed": s})
    if not data:
        sys.exit("Khong co file nao. Kiem tra --dir / --models / --seeds.")

    md = []

    def emit(x=""):
        print(x)
        md.append(x)

    for m, atks in data.items():
        n_list = sorted({x["n"] for rows in atks.values() for x in rows})
        emit(f"\n## {m} - so giao dich gia can de vuot detector (vi phishing ma mo hinh bat dung luc sach: {n_list} / seed)")
        emit("| Tan cong | quét tới | % vi vuot duoc | 10% vi | 25% vi | 50% vi |")
        emit("|---|---|---|---|---|---|")
        for atk in sorted(atks, key=ORDER.index):
            rows = atks[atk]
            fr = [sum(x is not None for x in r["min_txs"]) / max(r["n"], 1) for r in rows]
            q = {p: [quantile_txs(r["min_txs"], r["n"], p) for r in rows] for p in (0.10, 0.25, 0.50)}
            cell = lambda p: ("chua dat" if all(v is None for v in q[p]) else
                              (f"{np.mean([v for v in q[p] if v is not None]):.0f} tx" +
                               (f" ± {np.std([v for v in q[p] if v is not None]):.0f}" if len([v for v in q[p] if v is not None]) > 1 else "") +
                               (f" ({sum(v is None for v in q[p])}/{len(q[p])} seed chua dat)" if any(v is None for v in q[p]) else "")))
            emit(f"| {atk} | {rows[0]['cap']} tx | {100 * np.mean(fr):.1f}% | {cell(0.10)} | {cell(0.25)} | {cell(0.50)} |")

        emit(f"\n**{m}: ty le vi bi vuot khi dung toi da T giao dich gia**\n")
        emit("| Tan cong | " + " | ".join(f"≤ {T} tx" for T in CAPS) + " |")
        emit("|---|" + "---|" * len(CAPS))
        for atk in sorted(atks, key=ORDER.index):
            cells = []
            for T in CAPS:
                vals = []
                for r in atks[atk]:
                    if T > r["cap"]:
                        continue
                    vals.append(sum(x is not None and x <= T for x in r["min_txs"]) / max(r["n"], 1))
                cells.append("-" if not vals else
                             f"{100 * np.mean(vals):.1f}%" + (f" ± {100 * np.std(vals):.1f}" if len(vals) > 1 else ""))
            emit(f"| {atk} | " + " | ".join(cells) + " |")

    out = d / "mincost_summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
