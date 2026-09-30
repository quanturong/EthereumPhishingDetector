"""
Stage 4 – Normal test-only cho kịch bản mất cân bằng 1:5 / 1:10
================================================================
Theo dòng 77 đề cương: "dùng lại đúng nhóm phishing của tập test và bổ sung
normal để đạt tỷ lệ khoảng 1:5 (704 : 3.520) và, nếu điều kiện cho phép, 1:10.
Phần normal bổ sung này được thu ở cửa sổ thời gian sau T2."

Pipeline:
  4a. Xác định pool unused_valid (~15,944) từ neighbor_pool_checked
      -> crawl anchor_ts (1 API call/addr, ~66 phút)
  4b. Filter anchor_ts > T2 và stratified sample theo phân phối tx_count
      của tập test phishing hiện tại
  4c. Crawl full hop-1 txs cho các addr được chọn
  4d. Build ego-graph cutoff = T_end (test partition)
  4e. Assemble imbalanced test sets (1:5 và 1:10)

Output:
  stage4_dataset/anchor_ts_unused.csv
  stage4_dataset/normal_testonly_hop1_txs/*.json
  stage4_dataset/imbalanced_test_1to5.pkl
  stage4_dataset/imbalanced_test_1to10.pkl
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from crawl_normal_candidates import etherscan_get, DELAY
from stage3_pipeline import _build_one_ego_graph

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE1_DIR = BASE_DIR / "ethphishgraph2026_stage1_outputs"
STAGE2_DIR = BASE_DIR / "ethphishgraph2026_stage2_normal_crawl"
STAGE3_DIR = BASE_DIR / "ethphishgraph2026_stage3_dataset"
STAGE4_DIR = BASE_DIR / "ethphishgraph2026_stage4_imbalanced"
TESTONLY_HOP1_DIR = STAGE4_DIR / "normal_testonly_hop1_txs"

CHECKED_CSV   = STAGE2_DIR / "neighbor_pool_checked.csv"
NORMAL_CSV    = STAGE2_DIR / "normal_candidates.csv"
PHISHING_CSV  = STAGE1_DIR / "phishing_final_5_2000.csv"
SPLIT_CSV     = STAGE3_DIR / "temporal_split.csv"
SPLIT_META    = STAGE3_DIR / "temporal_split_meta.json"
DATASET_PKL   = STAGE3_DIR / "ego_graph_dataset.pkl"

ANCHOR_CSV       = STAGE4_DIR / "anchor_ts_unused.csv"
SELECTED_CSV     = STAGE4_DIR / "testonly_selected.csv"
IMBAL_5_PKL      = STAGE4_DIR / "imbalanced_test_1to5.pkl"
IMBAL_10_PKL     = STAGE4_DIR / "imbalanced_test_1to10.pkl"


# ── 4a. Fetch anchor_ts cho unused_valid ─────────────────────────────────────
def step_4a_fetch_anchor(api_key: str):
    checked = pd.read_csv(CHECKED_CSV)
    used    = set(pd.read_csv(NORMAL_CSV)["address"].str.lower())

    unused_valid = checked[
        (~checked["address"].isin(used)) &
        (checked["is_contract"] == False) &
        (checked["risk_labeled"] == False) &
        (checked["tx_count"] >= 5) &
        (checked["tx_count"] <= 2000)
    ].copy()
    unused_valid["address"] = unused_valid["address"].str.lower()
    print(f"[4a] Unused valid candidates: {len(unused_valid):,}")

    # Checkpoint
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[str, int] = {}
    if ANCHOR_CSV.exists() and ANCHOR_CSV.stat().st_size > 0:
        try:
            df = pd.read_csv(ANCHOR_CSV)
            done = dict(zip(df["address"], df["anchor_ts"]))
            print(f"[4a] Checkpoint: {len(done):,} đã có anchor_ts")
        except pd.errors.EmptyDataError:
            pass

    todo = [a for a in unused_valid["address"] if a not in done]
    print(f"[4a] Cần fetch thêm {len(todo):,} anchor_ts...")

    results = [{"address": a, "anchor_ts": ts} for a, ts in done.items()]
    BATCH = 300

    for i, addr in enumerate(tqdm(todo, unit="addr")):
        data = etherscan_get({
            "module": "account", "action": "txlist",
            "address": addr, "startblock": 0, "endblock": 99999999,
            "page": 1, "offset": 1, "sort": "asc",
        }, api_key)
        anchor = -1
        if data and data.get("status") == "1" and data["result"]:
            anchor = int(data["result"][0].get("timeStamp", -1))
        results.append({"address": addr, "anchor_ts": anchor})
        time.sleep(DELAY)

        if (i + 1) % BATCH == 0:
            pd.DataFrame(results).to_csv(ANCHOR_CSV, index=False)

    pd.DataFrame(results).to_csv(ANCHOR_CSV, index=False)
    print(f"[4a] Xong -> {ANCHOR_CSV}")


# ── 4b. Stratified sample post-T2 ─────────────────────────────────────────────
def step_4b_select():
    meta = json.loads(SPLIT_META.read_text(encoding="utf-8"))
    T2 = meta["T2_ts"]
    print(f"[4b] T2 = {meta['T2_iso']} ({T2})")

    anchor_df  = pd.read_csv(ANCHOR_CSV)
    checked_df = pd.read_csv(CHECKED_CSV)[["address", "tx_count"]]
    anchor_df = anchor_df.merge(checked_df, on="address", how="left")

    post_t2 = anchor_df[(anchor_df["anchor_ts"] > T2) & (anchor_df["anchor_ts"] > 0)].copy()
    print(f"[4b] Ứng viên có anchor_ts > T2: {len(post_t2):,}")

    # Stratify theo phân phối tx_count của tập test phishing
    split = pd.read_csv(SPLIT_CSV)
    test_phishing = split[(split["partition"] == "test") & (split["label"] == 1)]
    print(f"[4b] Test phishing count: {len(test_phishing)}")

    # Lấy tx_count của test phishing từ hop1_tx_count đã lưu
    bins = [5, 10, 20, 50, 100, 200, 500, 2001]
    bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    test_phishing = test_phishing.copy()
    test_phishing["tx_bin"] = pd.cut(test_phishing["hop1_tx_count"], bins=bins, labels=bin_labels, right=False)
    post_t2["tx_bin"] = pd.cut(post_t2["tx_count"], bins=bins, labels=bin_labels, right=False)

    bin_ratio = test_phishing["tx_bin"].value_counts(normalize=True).sort_index()

    # Cần: 594 × 10 = 5,940 normal cho 1:10; đã có 814 trong test partition
    # => cần bổ sung tối đa 5,126 addresses ở đây
    n_target_add = 5940 - 814  # ~5126
    print(f"[4b] Cần bổ sung tối đa {n_target_add:,} normal test-only")

    selected = []
    rng = np.random.default_rng(42)
    for bl in bin_labels:
        pool = post_t2[post_t2["tx_bin"] == bl]
        need = int(round(bin_ratio.get(bl, 0) * n_target_add))
        take = min(need, len(pool))
        if take == 0:
            continue
        idx = rng.choice(len(pool), size=take, replace=False)
        selected.append(pool.iloc[idx])
        print(f"  bin {bl}: cần {need}, pool={len(pool)}, lấy {take}")

    if not selected:
        raise RuntimeError("[4b] Không có ứng viên nào qua stratification.")
    sel_df = pd.concat(selected, ignore_index=True)
    sel_df.to_csv(SELECTED_CSV, index=False)
    print(f"[4b] Selected {len(sel_df):,} -> {SELECTED_CSV}")


# ── 4c. Full hop-1 crawl cho các addr đã chọn ────────────────────────────────
def step_4c_crawl_full_hop1(api_key: str):
    sel = pd.read_csv(SELECTED_CSV)
    addresses = sel["address"].str.lower().tolist()
    print(f"[4c] Crawl full hop-1 cho {len(addresses):,} test-only candidates...")
    TESTONLY_HOP1_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for addr in tqdm(addresses, unit="addr"):
        out = TESTONLY_HOP1_DIR / f"{addr}.json"
        if out.exists() and out.stat().st_size > 2:
            skipped += 1
            continue

        txs, page = [], 1
        while True:
            data = etherscan_get({
                "module": "account", "action": "txlist",
                "address": addr, "startblock": 0, "endblock": 99999999,
                "page": page, "offset": 10000, "sort": "asc",
            }, api_key)
            if data is None or data.get("status") != "1":
                break
            batch = data["result"]
            txs.extend(batch)
            if len(batch) < 10000:
                break
            page += 1
            time.sleep(DELAY)

        out.write_text(json.dumps(txs), encoding="utf-8")
        time.sleep(DELAY)
    print(f"[4c] Xong. Skipped {skipped}.")


# ── 4d + 4e. Build ego-graphs và assemble imbalanced sets ────────────────────
def step_4de_build_and_assemble():
    sel = pd.read_csv(SELECTED_CSV)
    print(f"[4de] Build ego-graphs cho {len(sel):,} test-only candidates...")

    # T_end = "vô cực" như bước 3c
    split = pd.read_csv(SPLIT_CSV)
    T_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600

    testonly = []
    n_dropped = 0
    for _, row in tqdm(sel.iterrows(), total=len(sel), unit="target"):
        addr = row["address"].lower()
        f = TESTONLY_HOP1_DIR / f"{addr}.json"
        if not f.exists() or f.stat().st_size <= 2:
            n_dropped += 1
            continue
        raw = json.loads(f.read_text(encoding="utf-8"))
        g = _build_one_ego_graph(addr, 0, T_end, raw)
        if g is None:
            n_dropped += 1
            continue
        g["partition"] = "test_only"
        testonly.append(g)

    print(f"[4de] Ego-graphs test-only: {len(testonly):,} (dropped {n_dropped})")

    # Load base test set
    with open(DATASET_PKL, "rb") as f:
        ds = pickle.load(f)
    test_phish = [g for g in ds["test"] if g["label"] == 1]
    test_norm  = [g for g in ds["test"] if g["label"] == 0]
    print(f"[4de] Base test set: {len(test_phish)} phish + {len(test_norm)} norm")

    def assemble(ratio: int) -> list[dict]:
        n_need_norm = len(test_phish) * ratio
        n_add = n_need_norm - len(test_norm)
        if n_add <= 0:
            add = []
        elif n_add >= len(testonly):
            print(f"  ⚠ Ratio 1:{ratio}: cần {n_add} thêm nhưng chỉ có {len(testonly)}. "
                  f"Ratio thực tế = 1:{(len(test_norm)+len(testonly))/len(test_phish):.2f}")
            add = testonly
        else:
            rng = np.random.default_rng(42 + ratio)
            idx = rng.choice(len(testonly), size=n_add, replace=False)
            add = [testonly[i] for i in idx]
        result = test_phish + test_norm + add
        n_phish = sum(g["label"] == 1 for g in result)
        n_norm  = sum(g["label"] == 0 for g in result)
        print(f"  1:{ratio} -> {len(result)} ego-graphs "
              f"(phish={n_phish}, norm={n_norm}, actual ratio 1:{n_norm/n_phish:.2f})")
        return result

    print("\n[4de] Assemble imbalanced test sets:")
    imb_5  = assemble(5)
    imb_10 = assemble(10)

    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    with open(IMBAL_5_PKL, "wb") as f:
        pickle.dump(imb_5, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(IMBAL_10_PKL, "wb") as f:
        pickle.dump(imb_10, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[4de] Saved:")
    print(f"  {IMBAL_5_PKL}")
    print(f"  {IMBAL_10_PKL}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 4 - imbalanced test-only")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--step", choices=["4a", "4b", "4c", "4de", "all"], default="all")
    args = parser.parse_args()

    STAGE4_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("4a", "all"):
        if not args.api_key:
            raise ValueError("Cần --api_key cho 4a")
        step_4a_fetch_anchor(args.api_key)
    if args.step in ("4b", "all"):
        step_4b_select()
    if args.step in ("4c", "all"):
        if not args.api_key:
            raise ValueError("Cần --api_key cho 4c")
        step_4c_crawl_full_hop1(args.api_key)
    if args.step in ("4de", "all"):
        step_4de_build_and_assemble()


if __name__ == "__main__":
    main()
