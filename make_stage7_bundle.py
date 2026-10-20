"""
Gom cac file can upload len Kaggle cho Stage 7 (phong thu nen peae_paper) vao 1 THU MUC, dat dung ten san.

Gom 3 file, dat ten dung:
  stage5_train_baselines.py   <- ban co PEAEGNNPaper (lay stage5_train_baselines.py neu da la ban moi, neu khong lay
                                 stage5_train_baselines_paper.py)
  stage7_defense.py           <- ban moi (co ra_peaepaper)
  feat_stats_peae.pt          <- D:\\NCKH\\Final\\results\\stage5_peae\\feat_stats.pt (thong ke dac trung cua peae_paper)
Kiem tra truoc khi dong goi (de khong upload nham ban cu):
  - stage5 co 'class PEAEGNNPaper' va 'add_peae_edge_stats'; stage7 co 'ra_peaepaper' va 'resolve_stats'
  - feat_stats co khoa 'peae_min' (dung la thong ke cua peae_paper, khong phai cua mo hinh khac)

Chay (trong thu muc code, bat venv):
  python make_stage7_bundle.py
Ket qua: thu muc D:\\NCKH\\Final\\kaggle_upload_peae chua 3 file - keo ca 3 vao hop Upload Data cua Kaggle
"""
import argparse
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent


def has(path: Path, *marks) -> bool:
    if not path.exists():
        return False
    t = path.read_text(encoding="utf-8", errors="ignore")
    return all(m in t for m in marks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", default=str(HERE))
    ap.add_argument("--stats", default=r"D:\NCKH\Final\results\stage5_peae\feat_stats.pt")
    ap.add_argument("--out", default=r"D:\NCKH\Final\kaggle_upload_peae")
    a = ap.parse_args()
    cd, out = Path(a.code_dir), Path(a.out)

    m5 = ("class PEAEGNNPaper", "add_peae_edge_stats")
    s5 = next((p for p in (cd / "stage5_train_baselines.py", cd / "stage5_train_baselines_paper.py") if has(p, *m5)), None)
    s7 = cd / "stage7_defense.py"
    errs = []
    if s5 is None:
        errs.append(f"KHONG THAY ban stage5 moi (can co {m5}) trong {cd}. Chep stage5_train_baselines_paper.py moi nhat vao day.")
    if not has(s7, "ra_peaepaper", "resolve_stats"):
        errs.append(f"{s7} la ban CU (thieu ra_peaepaper/resolve_stats). Chep stage7_defense.py moi nhat vao day.")
    stats = Path(a.stats)
    if not stats.exists():
        errs.append(f"KHONG THAY {stats}")
    else:
        import torch
        st = torch.load(stats, map_location="cpu", weights_only=True)
        if "peae_min" not in st:
            errs.append(f"{stats} KHONG phai thong ke cua peae_paper (thieu 'peae_min').")
    if errs:
        sys.exit("DUNG:\n  - " + "\n  - ".join(errs))

    out.mkdir(parents=True, exist_ok=True)
    files = {"stage5_train_baselines.py": s5, "stage7_defense.py": s7, "feat_stats_peae.pt": stats}
    for name, src in files.items():
        (out / name).write_bytes(src.read_bytes())
        print(f"  {name:<28s} {src.stat().st_size / 1000:7.2f} kB   (tu {src.name})")
    print(f"\nXONG. 3 file da dat dung ten trong thu muc:\n  {out}")
    print("\nTrong hop 'Upload Data' tren Kaggle:")
    print("  1) bam X canh stage5_train_baselines.py va stage7_defense.py cu (giu nguyen 2 file con lai)")
    print("  2) keo 3 file trong thu muc tren vao hop")
    print("  3) bam Create / Save version")
    print("Sau khi xong, stage7_defense.py phai khoang 31 kB (ban cu la 27.26 kB).")

if __name__ == "__main__":
    main()
