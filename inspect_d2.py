"""
Kiem tra nhanh cau truc that cua dataset XBlock D2 (khong sua gi, chi doc).
Chay:   python inspect_d2.py --d2 D:\\NCKH\\Final\\D2raw
(--d2 la thu muc chua / hoac nam tren 4 thu muc 钓鱼一阶节点, 钓鱼二阶节点, 非钓鱼一阶节点, 非钓鱼二阶节点;
 script tu tim trong toi da 3 cap thu muc con)
Gui toan bo phan in ra cho Claude.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES = {"ph1": "钓鱼一阶节点", "ph2": "钓鱼二阶节点", "nm1": "非钓鱼一阶节点", "nm2": "非钓鱼二阶节点"}


def find_root(base: Path):
    cands = [base] + [p for p in base.rglob("*") if p.is_dir() and len(p.relative_to(base).parts) <= 3]
    for c in cands:
        if all((c / n).is_dir() for n in NAMES.values()):
            return c
    return None


def read_csv(p: Path):
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(p, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    return None


def to_ts(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().mean() > 0.9:
        return s
    return pd.to_datetime(series, errors="coerce", utc=True).astype("int64") / 1e9


def fmt_ts(x):
    try:
        return datetime.fromtimestamp(float(x), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return str(x)


def describe_hop1(folder: Path, tag: str, max_files=4000):
    files = sorted(folder.glob("*.csv"))
    print(f"\n=== {tag}: {folder.name} ===")
    print(f"so file .csv truc tiep: {len(files)}  | so thu muc con: {sum(p.is_dir() for p in folder.iterdir())}")
    if not files:
        print("  (khong co csv truc tiep)")
        return []
    print("5 ten file dau:", [f.name for f in files[:5]])
    first = read_csv(files[0])
    if first is not None:
        print("cot:", list(first.columns), "| dong mau:")
        print(first.head(3).to_string(index=False))
        print("dtype:", dict(first.dtypes.astype(str)))
    n_rows, ts_min, ts_max, vals, self_ok = [], [], [], [], 0
    for f in files[:max_files]:
        df = read_csv(f)
        if df is None or df.empty or not {"TimeStamp", "Value", "From", "To"} <= set(df.columns):
            n_rows.append(0)
            continue
        n_rows.append(len(df))
        t = to_ts(df["TimeStamp"]).dropna()
        if len(t):
            ts_min.append(t.min()); ts_max.append(t.max())
        v = pd.to_numeric(df["Value"], errors="coerce").dropna()
        vals.append(v.sample(min(len(v), 50), random_state=0))
        addr = f.stem.lower()
        ends = set(df["From"].astype(str).str.lower()) | set(df["To"].astype(str).str.lower())
        self_ok += addr in ends
    nr = np.array(n_rows)
    print(f"so giao dich / file: min={nr.min()} trung vi={int(np.median(nr))} max={nr.max()} | file rong/thieu cot: {(nr == 0).sum()}")
    print(f"so file co dia chi (ten file) xuat hien trong From/To: {self_ok}/{len(files[:max_files])}")
    print(f"so file co tu 5 den 2000 giao dich: {((nr >= 5) & (nr <= 2000)).sum()}")
    if ts_min:
        print(f"TimeStamp som nhat: {fmt_ts(min(ts_min))} | muon nhat: {fmt_ts(max(ts_max))}")
    if vals:
        v = pd.concat(vals)
        pos = v[v > 0]
        print(f"Value (mau): min={v.min():.6g} trung vi(>0)={pos.median():.6g} p99={pos.quantile(.99):.6g} max={v.max():.6g} "
              f"| co phan thap phan: {bool(((v % 1) != 0).any())}")
        print("  -> neu trung vi ~ 1e15..1e18 thi la WEI; neu ~ 0.01..10 thi la ETH.")
    return [f.stem.lower() for f in files]


def describe_hop2(folder: Path, tag: str, hop1_names):
    subs = sorted(p for p in folder.iterdir() if p.is_dir())
    direct = sorted(folder.glob("*.csv"))
    print(f"\n=== {tag}: {folder.name} ===")
    print(f"so thu muc con (muc tieu): {len(subs)} | csv truc tiep: {len(direct)}")
    if hop1_names:
        names = {p.name.lower() for p in subs}
        print(f"ten thu muc con khop ten file bac 1: {len(names & set(hop1_names))}/{len(subs)}")
    for sd in subs[:3]:
        csvs = sorted(sd.glob("*.csv"))
        print(f"  - {sd.name[:20]}...: {len(csvs)} csv; ten dau: {[c.name[:14] for c in csvs[:3]]}")
        if csvs:
            df = read_csv(csvs[0])
            if df is not None:
                print("    cot:", list(df.columns), "| so dong:", len(df))
                print(df.head(2).to_string(index=False))
    if subs:
        counts = [len(list(sd.glob("*.csv"))) for sd in subs[:300]]
        print(f"so csv moi thu muc (300 dau): min={min(counts)} trung vi={int(np.median(counts))} max={max(counts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2", required=True)
    a = ap.parse_args()
    root = find_root(Path(a.d2))
    if root is None:
        sys.exit(f"Khong tim thay 4 thu muc D2 trong {a.d2} (toi da 3 cap thu muc con). "
                 f"Kiem tra duong dan / ten thu muc.")
    print(f"Thu muc goc D2: {root}")
    ph1 = describe_hop1(root / NAMES["ph1"], "PHISHING bac 1")
    nm1 = describe_hop1(root / NAMES["nm1"], "NORMAL bac 1")
    print(f"\n=== Trung lap giua phishing va normal (theo ten file): {len(set(ph1) & set(nm1))} dia chi ===")
    describe_hop2(root / NAMES["ph2"], "PHISHING bac 2", ph1)
    describe_hop2(root / NAMES["nm2"], "NORMAL bac 2", nm1)
    print("\nXong. Gui toan bo phan in ra o tren cho Claude.")


if __name__ == "__main__":
    main()
