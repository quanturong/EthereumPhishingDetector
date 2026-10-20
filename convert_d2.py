"""
Chuyen dataset XBlock D2 sang BO CUC THU MUC cua pipeline (de dung lai Stage 9/5/6/7 khong sua logic).

Dau ra (trong --out):
  ethphishgraph2026_stage2_normal_crawl/hop1_txs/<addr>.json       tx cua ca target PHISHING
  ethphishgraph2026_stage3_dataset/normal_hop1_txs/<addr>.json     tx cua ca target NORMAL
  ethphishgraph2026_stage9_k2/hop2_neighbor_txs/<neighbor>.json    tx cua cac hang xom bac 1 (tu thu muc bac 2)
  ethphishgraph2026_stage3_dataset/temporal_split.csv              address,label,partition,anchor_ts,tx_count
  ethphishgraph2026_stage3_dataset/temporal_split_meta.json        T1_ts, T2_ts, ...
  d2_conversion_report.json

Quy tac (dung nhu pipeline 2026):
  - Value D2 tinh bang ETH -> nhan 1e18 ra wei. D2 khong co gas -> dat gasPrice=20 gwei, gasUsed=21000 (gia).
    => cac mo hinh dung thong tin gas cua canh (ra_peae) KHONG dung cong bang tren D2.
  - Giu target co 5..2000 giao dich (so dong trong CSV bac 1, bo trung TxHash).
  - Mac dinh (--require_hop2) chi giu target CO thu muc bac 2: D2 thieu bac 2 khong deu giua 2 lop
    (phishing 76%, normal 94%), giu tat ca se tao dau hieu gia 'co/khong co bac 2'.
  - Bo dia chi 0x000...000 (dia chi huy, khong phai tai khoan thong thuong).
  - Moi hang xom giu toi da --max_neighbor_tx giao dich SOM NHAT (giong crawl 2026: sort=asc, 500).
  - Chia tap: --split temporal (mac dinh, 60/20/20 theo giao dich dau tien, cutoff T1/T2 nhu Stage 3)
              hoac random (phan tang theo nhan 60/20/20, KHONG cat theo thoi gian).

Chay:
  python convert_d2.py --d2 D:\\NCKH\\Final\\D2raw --out D:\\NCKH\\Final\\D2base --limit 40    # thu nhanh
  python convert_d2.py --d2 D:\\NCKH\\Final\\D2raw --out D:\\NCKH\\Final\\D2base
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES = {"ph1": "钓鱼一阶节点", "ph2": "钓鱼二阶节点", "nm1": "非钓鱼一阶节点", "nm2": "非钓鱼二阶节点"}
GAS_PRICE, GAS_USED = "20000000000", "21000"
ZERO_ADDR = "0x" + "0" * 40
FAR_FUTURE = 4102444800  # 2100-01-01


def read_csv(p: Path):
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(p, encoding=enc, on_bad_lines="skip", low_memory=False,
                               usecols=lambda c: c in {"TxHash", "TimeStamp", "From", "To", "Value", "isError"})
        except Exception:
            continue
    return None


def csv_to_txs(p: Path):
    df = read_csv(p)
    if df is None or df.empty or not {"TimeStamp", "From", "To", "Value"} <= set(df.columns):
        return []
    ts = pd.to_numeric(df["TimeStamp"], errors="coerce")
    ok = ts.notna() & (ts > 0)
    df, ts = df[ok], ts[ok].astype("int64")
    if df.empty:
        return []
    val = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frm = df["From"].fillna("").astype(str).str.lower()
    to = df["To"].fillna("").astype(str).str.lower()
    h = df["TxHash"].fillna("").astype(str) if "TxHash" in df else pd.Series([""] * len(df), index=df.index)
    err = (pd.to_numeric(df["isError"], errors="coerce").fillna(0).astype(int).astype(str)
           if "isError" in df else pd.Series(["0"] * len(df), index=df.index))
    txs, seen = [], set()
    for f, t, v, s, hh, e in zip(frm, to, val, ts, h, err):
        if hh:
            if hh in seen:
                continue
            seen.add(hh)
        txs.append({"from": f, "to": t, "value": str(int(v * 1e18)), "timeStamp": str(int(s)),
                    "gasPrice": GAS_PRICE, "gasUsed": GAS_USED, "isError": e, "hash": hh})
    txs.sort(key=lambda x: int(x["timeStamp"]))
    return txs


def merge(a, b):
    seen, out = set(), []
    for t in a + b:
        k = t["hash"] or (t["from"], t["to"], t["value"], t["timeStamp"])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    out.sort(key=lambda x: int(x["timeStamp"]))
    return out


def find_root(base: Path):
    for c in [base] + [p for p in base.rglob("*") if p.is_dir() and len(p.relative_to(base).parts) <= 3]:
        if all((c / n).is_dir() for n in NAMES.values()):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_tx", type=int, default=5)
    ap.add_argument("--max_tx", type=int, default=2000)
    ap.add_argument("--max_neighbor_tx", type=int, default=500)
    ap.add_argument("--no_require_hop2", action="store_true", help="giu ca target khong co thu muc bac 2")
    ap.add_argument("--split", choices=["temporal", "random"], default="temporal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="chi lay N target dau moi lop (thu nhanh)")
    a = ap.parse_args()

    root = find_root(Path(a.d2))
    if root is None:
        sys.exit(f"Khong tim thay 4 thu muc D2 trong {a.d2}")
    out = Path(a.out)
    d_ph = out / "ethphishgraph2026_stage2_normal_crawl" / "hop1_txs"
    d_s3 = out / "ethphishgraph2026_stage3_dataset"
    d_nm = d_s3 / "normal_hop1_txs"
    d_h2 = out / "ethphishgraph2026_stage9_k2" / "hop2_neighbor_txs"
    for d in (d_ph, d_nm, d_h2):
        d.mkdir(parents=True, exist_ok=True)

    report = {"settings": vars(a), "per_class": {}}
    targets = []   # (addr, label, anchor_ts, tx_count)
    for label, h1n, h2n in ((1, NAMES["ph1"], NAMES["ph2"]), (0, NAMES["nm1"], NAMES["nm2"])):
        files = sorted((root / h1n).glob("*.csv"))
        if a.limit:
            files = files[: a.limit]
        drop = Counter()
        for f in tqdm(files, desc=f"bac 1 {'phishing' if label else 'normal'}", unit="file"):
            addr = f.stem.lower()
            if addr == ZERO_ADDR:
                drop["zero_address"] += 1
                continue
            txs = csv_to_txs(f)
            if not txs:
                drop["empty"] += 1
                continue
            if len(txs) < a.min_tx:
                drop["lt_min_tx"] += 1
                continue
            if len(txs) > a.max_tx:
                drop["gt_max_tx"] += 1
                continue
            h2dir = root / h2n / f.stem
            if not a.no_require_hop2 and not (h2dir.is_dir() and any(h2dir.glob("*.csv"))):
                drop["no_hop2"] += 1
                continue
            (d_ph if label else d_nm).joinpath(f"{addr}.json").write_text(json.dumps(txs), encoding="utf-8")
            targets.append((addr, label, int(txs[0]["timeStamp"]), len(txs), h2dir))
        report["per_class"]["phishing" if label else "normal"] = {
            "raw_files": len(files), "kept": sum(1 for t in targets if t[1] == label), "dropped": dict(drop)}

    # hang xom bac 1 (tu thu muc bac 2)
    written, n_rows = {}, 0
    for addr, label, _, _, h2dir in tqdm(targets, desc="bac 2", unit="target"):
        if not h2dir.is_dir():
            continue
        for c in h2dir.glob("*.csv"):
            nb = c.stem.lower()
            txs = csv_to_txs(c)
            if not txs:
                continue
            p = d_h2 / f"{nb}.json"
            if nb in written:
                txs = merge(json.loads(p.read_text(encoding="utf-8")), txs)
            txs = txs[: a.max_neighbor_tx]
            p.write_text(json.dumps(txs), encoding="utf-8")
            written[nb] = True
            n_rows += len(txs)

    # chia tap
    df = pd.DataFrame([(t[0], t[1], t[2], t[3]) for t in targets], columns=["address", "label", "anchor_ts", "tx_count"])
    n = len(df)
    if n < 10:
        sys.exit(f"Chi con {n} target sau loc - qua it de chia tap (dung --limit lon hon).")
    if a.split == "temporal":
        df = df.sort_values("anchor_ts").reset_index(drop=True)
        i1, i2 = int(n * 0.6), int(n * 0.8)
        df["partition"] = ["train"] * i1 + ["val"] * (i2 - i1) + ["test"] * (n - i2)
        T1, T2 = int(df.loc[i1, "anchor_ts"]), int(df.loc[i2, "anchor_ts"])
    else:
        rng = np.random.default_rng(a.seed)
        df["partition"] = ""
        for lab in (0, 1):
            idx = df.index[df["label"] == lab].to_numpy()
            rng.shuffle(idx)
            k1, k2 = int(len(idx) * 0.6), int(len(idx) * 0.8)
            df.loc[idx[:k1], "partition"] = "train"
            df.loc[idx[k1:k2], "partition"] = "val"
            df.loc[idx[k2:], "partition"] = "test"
        T1 = T2 = FAR_FUTURE        # khong cat theo thoi gian
    df[["address", "label", "partition", "anchor_ts", "tx_count"]].to_csv(d_s3 / "temporal_split.csv", index=False)
    (d_s3 / "temporal_split_meta.json").write_text(
        json.dumps({"T1_ts": T1, "T2_ts": T2, "split": a.split}, indent=2), encoding="utf-8")

    iso = lambda x: pd.to_datetime(x, unit="s").strftime("%Y-%m-%d")
    report.update({"n_targets": n, "n_neighbor_files": len(written), "n_neighbor_rows": n_rows,
                   "T1": iso(T1) if T1 < FAR_FUTURE else None, "T2": iso(T2) if T2 < FAR_FUTURE else None,
                   "partition_by_label": {p: {"phishing": int(((df.partition == p) & (df.label == 1)).sum()),
                                              "normal": int(((df.partition == p) & (df.label == 0)).sum())}
                                          for p in ("train", "val", "test")}})
    (out / "d2_conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ KET QUA CHUYEN DOI ================")
    for k, v in report["per_class"].items():
        print(f"{k:9s}: {v['raw_files']} file -> giu {v['kept']} | bo: {v['dropped']}")
    print(f"Tong target: {n} | file hang xom: {len(written)} | cach chia: {a.split} (T1={report['T1']}, T2={report['T2']})")
    for p, v in report["partition_by_label"].items():
        tot = v["phishing"] + v["normal"]
        print(f"  {p:5s}: {tot:5d}  phishing={v['phishing']:5d} ({100 * v['phishing'] / max(tot, 1):.0f}%)  normal={v['normal']:5d}")
    print(f"\nDa ghi vao: {out}\nBao cao: {out / 'd2_conversion_report.json'}")


if __name__ == "__main__":
    main()
