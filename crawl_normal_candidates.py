"""
Stage 2 – Crawl normal candidates cho EthPhishGraph-2026
=========================================================
Pipeline:
  2a. Lấy tất cả giao dịch hop-1 của 3518 phishing address (Etherscan API)
  2b. Gom tất cả địa chỉ neighbor -> pool ứng viên thô
  2c. Lọc: bỏ phishing, bỏ risk-labeled, bỏ contract, giữ tx ∈ [5, 2000]
  2d. Stratified sampling theo phân phối tx_count của tập phishing
      -> ~3518 normal candidates

Cách dùng:
  python crawl_normal_candidates.py --api_key YOUR_KEY
  python crawl_normal_candidates.py --api_key YOUR_KEY --step 2b  # chỉ chạy từ bước 2b

Checkpoint: mỗi địa chỉ phishing lưu file riêng trong hop1_txs/
  -> có thể Ctrl+C và chạy lại, sẽ bỏ qua địa chỉ đã crawl.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\EthereumPhishingDetection\dataset\EthPhishGraph-2026")
STAGE1_DIR = BASE_DIR / "ethphishgraph2026_stage1_outputs"
STAGE2_DIR = BASE_DIR / "ethphishgraph2026_stage2_normal_crawl"
HOP1_DIR   = STAGE2_DIR / "hop1_txs"

PHISHING_CSV   = STAGE1_DIR / "phishing_final_5_2000.csv"
POOL_CSV       = STAGE2_DIR / "neighbor_pool_raw.csv"
CHECKED_CSV    = STAGE2_DIR / "neighbor_pool_checked.csv"
NORMAL_OUT_CSV = STAGE2_DIR / "normal_candidates.csv"

# ── Etherscan API V2 ──────────────────────────────────────────────────────────
ETHERSCAN_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1          # Ethereum mainnet
DELAY = 0.25          # 4 req/sec (free tier = 5/sec, để dư)
MAX_RETRIES = 5


def etherscan_get(params: dict, api_key: str) -> dict | None:
    """
    Gọi Etherscan V2. Xử lý cả 2 format response:
      - REST (account/txlist): {"status": "1", "message": "OK", "result": [...]}
      - JSON-RPC (proxy/eth_getCode): {"jsonrpc": "2.0", "id": 1, "result": "0x..."}
    """
    params["apikey"] = api_key
    params["chainid"] = CHAIN_ID
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(ETHERSCAN_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            # JSON-RPC response (proxy module): không có "status"
            if "jsonrpc" in data:
                # Rate-limit qua field "error" hoặc "result" text
                err = data.get("error") or {}
                if err and "rate" in str(err.get("message", "")).lower():
                    time.sleep(2 ** attempt)
                    continue
                return data

            # REST response (module=account,...)
            status = data.get("status")
            if status == "1":
                return data
            if "Max rate limit reached" in str(data.get("result", "")):
                time.sleep(2 ** attempt)
                continue
            return data
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


# ── Bước 2a: crawl hop-1 txs ─────────────────────────────────────────────────
def step_2a_crawl_hop1(api_key: str):
    """Lấy toàn bộ giao dịch thường (normal tx) của từng phishing address."""
    phishing_df = pd.read_csv(PHISHING_CSV)
    addresses = phishing_df["address"].str.lower().tolist()

    print(f"[2a] Crawl hop-1 txs cho {len(addresses)} phishing addresses...")
    HOP1_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for addr in tqdm(addresses, unit="addr"):
        out_file = HOP1_DIR / f"{addr}.json"
        if out_file.exists():          # checkpoint: đã crawl rồi
            skipped += 1
            continue

        txs = []
        page = 1
        while True:
            params = {
                "module":     "account",
                "action":     "txlist",
                "address":    addr,
                "startblock": 0,
                "endblock":   99999999,
                "page":       page,
                "offset":     10000,   # max per call
                "sort":       "asc",
            }
            data = etherscan_get(params, api_key)
            if data is None or data["status"] != "1":
                break
            batch = data["result"]
            txs.extend(batch)
            if len(batch) < 10000:
                break
            page += 1
            time.sleep(DELAY)

        out_file.write_text(json.dumps(txs), encoding="utf-8")
        time.sleep(DELAY)

    print(f"[2a] Xong. Bỏ qua {skipped} địa chỉ đã có checkpoint.")


# ── Bước 2b: gom neighbor pool ───────────────────────────────────────────────
def step_2b_build_pool():
    """Gom tất cả địa chỉ from/to trong hop-1 txs, loại phishing ngay."""
    phishing_set = set(
        pd.read_csv(PHISHING_CSV)["address"].str.lower()
    )

    print("[2b] Gom neighbor pool từ hop-1 txs...")
    addr_set: set[str] = set()

    tx_files = list(HOP1_DIR.glob("*.json"))
    for f in tqdm(tx_files, unit="file"):
        txs = json.loads(f.read_text(encoding="utf-8"))
        for tx in txs:
            sender = tx.get("from", "").lower()
            recv   = tx.get("to",   "").lower()
            if sender and sender not in phishing_set:
                addr_set.add(sender)
            if recv and recv not in phishing_set:
                addr_set.add(recv)

    # Loại bỏ chính các phishing address (phòng trường hợp trùng)
    addr_set -= phishing_set

    pool_df = pd.DataFrame({"address": sorted(addr_set)})
    pool_df.to_csv(POOL_CSV, index=False)
    print(f"[2b] Pool thô: {len(pool_df):,} địa chỉ -> {POOL_CSV}")


# ── Bước 2c: kiểm tra từng ứng viên ──────────────────────────────────────────
def _is_contract(addr: str, api_key: str) -> bool:
    """True nếu là smart contract (code != '0x')."""
    data = etherscan_get({
        "module":  "proxy",
        "action":  "eth_getCode",
        "address": addr,
        "tag":     "latest",
    }, api_key)
    if data and data.get("result") not in ("0x", None, ""):
        return True
    return False


def _get_tx_count(addr: str, api_key: str) -> int | None:
    """Trả về số giao dịch thường (txlist). None nếu lỗi."""
    params = {
        "module":     "account",
        "action":     "txlist",
        "address":    addr,
        "startblock": 0,
        "endblock":   99999999,
        "page":       1,
        "offset":     1,          # chỉ cần tổng số, dùng trick page
        "sort":       "asc",
    }
    # Etherscan không trả tổng số trực tiếp -> lấy page cuối
    # Cách nhanh: getTransactionCount qua proxy (tính cả pending, không chính xác)
    # -> dùng txlist page lớn rồi đếm
    data = etherscan_get({
        "module":     "account",
        "action":     "txlist",
        "address":    addr,
        "startblock": 0,
        "endblock":   99999999,
        "page":       1,
        "offset":     10000,
        "sort":       "asc",
    }, api_key)
    if data is None:
        return None
    if data["status"] == "0":
        return 0
    return len(data["result"])


def step_2c_check_candidates(
    api_key: str,
    risk_labels_csv: str | None = None,
    pool_sample_size: int | None = None,
    seed: int = 42,
):
    """
    Với mỗi ứng viên trong pool:
      - Đếm tx (lọc [5, 2000]) qua 1 API call
      - Nếu qua lọc tx -> check contract (thêm 1 API call)
      - Loại risk-labeled nếu có file nhãn
    Lưu checkpoint từng batch 200 địa chỉ vào CHECKED_CSV.

    pool_sample_size: nếu được đặt, random-sample ngần đó địa chỉ từ pool
    trước khi check (giảm thời gian; đủ để lấy ~3518 normal candidates sau lọc).
    """
    pool_df = pd.read_csv(POOL_CSV)
    if pool_sample_size and pool_sample_size < len(pool_df):
        pool_df = pool_df.sample(n=pool_sample_size, random_state=seed).reset_index(drop=True)
        pool_df.to_csv(POOL_CSV.with_name("neighbor_pool_sampled.csv"), index=False)
        print(f"[2c] Random sample pool: {pool_sample_size:,} địa chỉ")
    addresses = pool_df["address"].str.lower().tolist()

    # Load risk labels nếu có
    risk_set: set[str] = set()
    if risk_labels_csv and Path(risk_labels_csv).exists():
        risk_df = pd.read_csv(risk_labels_csv)
        col = [c for c in risk_df.columns if "address" in c.lower()][0]
        risk_set = set(risk_df[col].str.lower())
        print(f"[2c] Loaded {len(risk_set):,} risk-labeled addresses từ {risk_labels_csv}")

    # Load checkpoint nếu có (guard empty file)
    done: dict[str, dict] = {}
    if CHECKED_CSV.exists() and CHECKED_CSV.stat().st_size > 0:
        try:
            existing = pd.read_csv(CHECKED_CSV)
            for _, row in existing.iterrows():
                done[row["address"]] = row.to_dict()
            print(f"[2c] Checkpoint: {len(done):,} địa chỉ đã kiểm tra")
        except pd.errors.EmptyDataError:
            print("[2c] Checkpoint file rỗng, bỏ qua.")

    todo = [a for a in addresses if a not in done]
    print(f"[2c] Cần kiểm tra thêm {len(todo):,} địa chỉ...")

    results = list(done.values())
    BATCH = 200

    # Thứ tự: tx_count trước (1 API call) -> chỉ check contract nếu tx ∈ [5,2000]
    # -> tiết kiệm ~50-70% API call so với check cả hai luôn.
    for i, addr in enumerate(tqdm(todo, unit="addr")):
        rec: dict = {"address": addr, "is_contract": None, "tx_count": None,
                     "risk_labeled": addr in risk_set}

        if addr in risk_set:
            rec["is_contract"] = False
            rec["tx_count"] = -1
        else:
            tx_cnt = _get_tx_count(addr, api_key)
            rec["tx_count"] = tx_cnt if tx_cnt is not None else -1
            time.sleep(DELAY)

            # Chỉ tốn thêm 1 API call cho những địa chỉ có khả năng qua lọc
            if tx_cnt is not None and 5 <= tx_cnt <= 2000:
                rec["is_contract"] = _is_contract(addr, api_key)
                time.sleep(DELAY)
            else:
                rec["is_contract"] = False  # placeholder; sẽ bị loại bởi tx filter

        results.append(rec)

        if (i + 1) % BATCH == 0:
            pd.DataFrame(results).to_csv(CHECKED_CSV, index=False)

    pd.DataFrame(results).to_csv(CHECKED_CSV, index=False)
    print(f"[2c] Xong -> {CHECKED_CSV}")


# ── Bước 2d: stratified sampling ─────────────────────────────────────────────
def step_2d_sample(target_n: int = 3518, seed: int = 42):
    """
    Lọc pool đã kiểm tra và lấy mẫu phân tầng theo phân phối
    tx_count của tập phishing.
    """
    phishing_df = pd.read_csv(PHISHING_CSV)
    if not CHECKED_CSV.exists() or CHECKED_CSV.stat().st_size == 0:
        raise FileNotFoundError(
            f"[2d] Chưa có {CHECKED_CSV.name}. Chạy bước 2c trước."
        )
    checked_df  = pd.read_csv(CHECKED_CSV)

    # Lọc ứng viên hợp lệ
    valid = checked_df[
        (checked_df["is_contract"] == False) &
        (checked_df["risk_labeled"] == False) &
        (checked_df["tx_count"] >= 5) &
        (checked_df["tx_count"] <= 2000)
    ].copy()
    print(f"[2d] Ứng viên hợp lệ sau lọc: {len(valid):,}")

    # Bins theo phân phối phishing
    bins = [5, 10, 20, 50, 100, 200, 500, 2001]
    bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]

    phishing_df["tx_bin"] = pd.cut(
        phishing_df["tx_count_capped"], bins=bins, labels=bin_labels, right=False
    )
    valid["tx_bin"] = pd.cut(
        valid["tx_count"], bins=bins, labels=bin_labels, right=False
    )

    bin_counts = phishing_df["tx_bin"].value_counts().sort_index()
    print("\n[2d] Phân phối tx_count của phishing (mục tiêu sampling):")
    print(bin_counts.to_string())

    sampled_parts = []
    rng = np.random.default_rng(seed)

    for bin_label in bin_labels:
        n_needed = int(bin_counts.get(bin_label, 0))
        pool_bin = valid[valid["tx_bin"] == bin_label]

        if len(pool_bin) == 0:
            print(f"  ⚠ Bin {bin_label}: cần {n_needed}, pool rỗng -> bỏ qua")
            continue
        if len(pool_bin) < n_needed:
            print(f"  ⚠ Bin {bin_label}: cần {n_needed}, có {len(pool_bin)} -> lấy hết")
            sampled_parts.append(pool_bin)
        else:
            idx = rng.choice(len(pool_bin), size=n_needed, replace=False)
            sampled_parts.append(pool_bin.iloc[idx])

    normal_df = pd.concat(sampled_parts, ignore_index=True)
    normal_df = normal_df[["address", "tx_count", "tx_bin"]].copy()
    normal_df["label"] = 0   # normal = 0

    normal_df.to_csv(NORMAL_OUT_CSV, index=False)
    print(f"\n[2d] Normal candidates: {len(normal_df):,} -> {NORMAL_OUT_CSV}")
    print(normal_df["tx_bin"].value_counts().sort_index().to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Crawl normal candidates - EthPhishGraph-2026")
    parser.add_argument("--api_key", required=True, help="Etherscan API key")
    parser.add_argument(
        "--step",
        choices=["2a", "2b", "2c", "2d", "all"],
        default="all",
        help="Chạy bước cụ thể hoặc toàn bộ (default: all)",
    )
    parser.add_argument(
        "--risk_labels",
        default=None,
        help="(Tuỳ chọn) CSV chứa cột address của các địa chỉ risk-labeled",
    )
    parser.add_argument("--target_n", type=int, default=3518,
                        help="Số normal candidates cần lấy (default: 3518)")
    parser.add_argument("--pool_sample_size", type=int, default=None,
                        help="(Tuỳ chọn) Random sample pool xuống size này trước khi check "
                             "(khuyến nghị 30000 để rút gọn thời gian còn ~2-3 giờ)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("2a", "all"):
        step_2a_crawl_hop1(args.api_key)
    if args.step in ("2b", "all"):
        step_2b_build_pool()
    if args.step in ("2c", "all"):
        step_2c_check_candidates(
            args.api_key, args.risk_labels,
            pool_sample_size=args.pool_sample_size, seed=args.seed,
        )
    if args.step in ("2d", "all"):
        step_2d_sample(args.target_n, args.seed)


if __name__ == "__main__":
    main()
