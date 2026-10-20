"""
Chuan bi + (tuy chon) upload dataset 'ethphish-pool' len Kaggle cho Stage 7.

Lam 2 viec:
  Buoc 1: gom 4 file vao 1 thu muc (D:\\NCKH\\Final\\kaggle_upload) va tao dataset-metadata.json
  Buoc 2: upload len Kaggle bang Kaggle CLI (neu da cai kaggle + co kaggle.json)

Dat file nay CUNG THU MUC voi stage5_train_baselines.py va stage7_defense.py
(D:\\NCKH\\Final\\EthereumPhishingDetector), roi chay:

    python prepare_kaggle_upload.py                 # gom file + upload
    python prepare_kaggle_upload.py --no_upload     # chi gom file (tu upload bang web)
    python prepare_kaggle_upload.py --update        # dataset DA CO -> tao phien ban moi

Neu chua co Kaggle CLI / kaggle.json, script se gom file xong va chi cho ban cach lam tiep.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "adv_pool_train.pkl": Path(r"D:\NCKH\Final\ethphishgraph2026_stage7_defense_k2\adv_pool_train.pkl"),
    "feat_stats.pt": Path(r"D:\NCKH\Final\results\stage5_k2\feat_stats.pt"),
    "stage5_train_baselines.py": HERE / "stage5_train_baselines.py",
    "stage7_defense.py": HERE / "stage7_defense.py",
}
# Dau hieu cua ban stage7_defense.py MOI (co --methods/--seeds va nen PEAE-style)
STAGE7_MARKERS = ['"--methods"', '"--seeds"', "ra_peae", "ADAPTIVE-STC-LV"]


def credentials_ok() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.exists()


def kaggle_cmd():
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    try:
        import kaggle  # noqa: F401
        return [sys.executable, "-c", "from kaggle.cli import main; main()"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"D:\NCKH\Final\kaggle_upload")
    ap.add_argument("--username", default="minhquan21007", help="ten tai khoan Kaggle (trong duong dan /kaggle/input/datasets/<ten>/...)")
    ap.add_argument("--slug", default="ethphish-pool")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--update", action="store_true", help="dataset da ton tai: tao phien ban moi thay vi tao moi")
    ap.add_argument("--msg", default="update")
    args = ap.parse_args()

    dest = Path(args.dest)
    print("== Buoc 1: kiem tra va gom file ==")
    missing = [n for n, p in DEFAULTS.items() if not p.exists()]
    if missing:
        for n in missing:
            print(f"  [THIEU] {n}  <- {DEFAULTS[n]}")
        sys.exit("Thieu file o tren. Kiem tra duong dan / chep file vao dung cho roi chay lai.")

    s7 = DEFAULTS["stage7_defense.py"].read_text(encoding="utf-8", errors="ignore")
    absent = [m for m in STAGE7_MARKERS if m not in s7]
    if absent:
        sys.exit(f"stage7_defense.py trong {DEFAULTS['stage7_defense.py'].parent} la BAN CU "
                 f"(thieu {absent}). Chep ban moi (tai tu tin nhan cua Claude) de de len roi chay lai.")

    dest.mkdir(parents=True, exist_ok=True)
    for name, src in DEFAULTS.items():
        out = dest / name
        if out.exists() and out.stat().st_size == src.stat().st_size and out.stat().st_mtime >= src.stat().st_mtime:
            print(f"  [giu nguyen] {name} ({out.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  [chep] {name} ({src.stat().st_size / 1e6:.1f} MB) ...")
        shutil.copy2(src, out)

    meta = {"title": args.slug, "id": f"{args.username}/{args.slug}", "licenses": [{"name": "CC0-1.0"}]}
    (dest / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Thu muc san sang: {dest}\n  Cac file: {sorted(p.name for p in dest.iterdir())}")

    if args.no_upload:
        print("\nDa chon --no_upload. Upload bang web: kaggle.com -> Datasets -> New Dataset, keo 4 file "
              f"(khong can dataset-metadata.json) tu {dest}, dat ten {args.slug}, Private.")
        return

    print("\n== Buoc 2: upload len Kaggle ==")
    cmd = kaggle_cmd()
    if cmd is None:
        sys.exit("Chua cai Kaggle CLI. Chay:  pip install kaggle   roi chay lai script (hoac upload bang web, "
                 "xem --no_upload).")
    if not credentials_ok():
        sys.exit("Chua co thong tin dang nhap Kaggle. Lam: kaggle.com -> anh dai dien -> Settings -> API -> "
                 "'Create New Token' (tai kaggle.json), chep vao C:\\Users\\<ten-ban>\\.kaggle\\kaggle.json, "
                 "roi chay lai. (Hoac upload bang web: --no_upload.)")

    sub = ["datasets", "version", "-p", str(dest), "-m", args.msg] if args.update \
        else ["datasets", "create", "-p", str(dest)]
    print("  Chay:", " ".join(sub), "(file ~1 GB, mat vai phut den vai chuc phut)")
    r = subprocess.run(cmd + sub)
    if r.returncode != 0:
        sys.exit("\nUpload that bai (xem thong bao o tren). Cach thuong gap: dataset da ton tai -> them --update; "
                 "loi dang nhap -> kiem tra kaggle.json. Co the upload bang web (--no_upload).")
    print(f"\nXong. Dataset: https://www.kaggle.com/datasets/{args.username}/{args.slug}  (Private)")
    print("Doi Kaggle xu ly xong (vai phut) roi them vao notebook. Chay cell os.walk de xem duong dan that.")


if __name__ == "__main__":
    main()
