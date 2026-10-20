"""
Dong goi KET QUA khoa luan vao 1 file .zip de gui di (nho, khong co du lieu tho).

Co trong goi:
  code/                     cac file .py (khong gom venv, __pycache__)
  stage5_baselines/         results_all.json, feat_stats.pt, checkpoint 3 mo hinh x 3 seed
  stage6_attacks/           results_*.json (moi lan chay tan cong), stage6_summary*.md/.csv, diag_feature_*.json
  stage7_defense/           eval_*.json, eval_probs_*.npz, stage7_summary/auc/stats *.md/.csv,
                            train_*.json va checkpoint cac mo hinh phong thu
  stage8_D2/                stage8_zeroshot_d2.json, d2_conversion_report.json
  README.txt, MANIFEST.txt (kich thuoc + SHA-256), requirements_freeze.txt

KHONG co (qua nang / khong nen gui): du lieu giao dich tho, ego_graph_*.pkl (vai GB), adv_pool_train.pkl (~1 GB),
  file thu 1 epoch / --smoke, venv, khoa API (script tu quet va BO file nghi co khoa).

Chay (o thu muc code):
  python package_results.py
  python package_results.py --root D:\\NCKH\\Final --out D:\\NCKH\\Final\\ketqua_klt.zip
  python package_results.py --no_models        # bo cac file .pt (nho hon)
"""
import argparse
import hashlib
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SECRET_RE = re.compile(
    r"""(?ix)
    ( (api[_-]?key|apikey|secret|passwd|password|token|access[_-]?key)\s*[=:]\s*['"][A-Za-z0-9_\-]{16,}['"]
    | default\s*=\s*['"][A-Z0-9]{30,40}['"]         # vd default='ABCDEF...' (khoa Etherscan 34 ky tu)
    | ["']username["']\s*:\s*["'][^"']+["']\s*,\s*["']key["']
    | KGAT_[A-Za-z0-9]{10,} )
    """)
TEXT_EXT = {".py", ".json", ".md", ".txt", ".csv", ".yml", ".yaml", ".cfg", ".ini"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_secret(p: Path) -> str | None:
    if p.suffix.lower() not in TEXT_EXT or p.stat().st_size > 5_000_000:
        return None
    try:
        m = SECRET_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return m.group(0)[:60] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\NCKH\Final")
    ap.add_argument("--code_dir", default=None, help="mac dinh: thu muc chua script nay")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_models", action="store_true", help="bo cac file checkpoint .pt")
    ap.add_argument("--no_code", action="store_true")
    ap.add_argument("--max_mb", type=float, default=50.0, help="bo file don le lon hon gioi han nay")
    ap.add_argument("--allow_secrets", action="store_true", help="KHONG khuyen dung")
    ap.add_argument("--extra", nargs="*", default=[], help="them file vao goi (vd file .md, .docx), dat trong thu muc tai_lieu/")
    a = ap.parse_args()

    root = Path(a.root)
    code_dir = Path(a.code_dir) if a.code_dir else Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(a.out) if a.out else root / f"ketqua_khoaluan_{stamp}.zip"

    s5 = root / "results" / "stage5_k2"
    s5v2 = root / "results" / "stage5_k2_7_10"      # peaegnnv2
    s7t = root / "results" / "stage7_k2"
    s6 = root / "ethphishgraph2026_stage6_attacks_k2"
    s7 = root / "ethphishgraph2026_stage7_defense_k2"
    d2 = root / "D2base"

    entries, missing = [], []   # (ten trong zip, duong dan)

    def add(group, folder, patterns, exclude=(), models=False):
        found = 0
        for pat in patterns:
            for p in sorted(folder.glob(pat)):
                if not p.is_file() or any(x in p.name for x in exclude):
                    continue
                m = re.search(r"_sh\d+of\d+(\.json)$", p.name)       # manh con sot lai khi da co file chinh
                if m and (p.parent / re.sub(r"_sh\d+of\d+(\.json)$", r"\1", p.name)).exists():
                    continue
                if p.suffix == ".pt" and a.no_models:
                    continue
                sub = "models/" if p.suffix == ".pt" and p.name.startswith("model_") else ""
                entries.append((f"KetQua_KLTN/{group}/{sub}{p.name}", p))
                found += 1
        if found == 0:
            missing.append(f"{group}  <-  {folder}  ({', '.join(patterns)})")
        return found

    add("stage5_baselines", s5, ["results_all.json", "feat_stats.pt", "model_*_seed*.pt"])
    add("stage5_peaegnnv2", s5v2, ["results_peaegnnv2.json", "feat_stats.pt", "model_peaegnnv2_seed*.pt"])
    add("stage6_attacks", s6, ["*.json", "*.md", "*.csv"], exclude=("_try", "smoke"))   # bo ket qua chay thu
    add("stage7_defense", s7, ["eval_*.json", "eval_probs_*.npz", "stage7_*.md", "stage7_*.csv"])
    add("stage7_defense", s7t, ["train_*.json", "model_*_seed*.pt"], exclude=("smoke",))
    add("stage8_D2", d2, ["stage8_zeroshot_d2.json", "d2_conversion_report.json"])
    if (code_dir / "d2_report.txt").exists():       # ket qua inspect_d2.py (cau truc/don vi cua D2)
        entries.append(("KetQua_KLTN/stage8_D2/d2_report.txt", code_dir / "d2_report.txt"))
    if not a.no_code:
        n = 0
        for p in sorted(code_dir.glob("*.py")):
            entries.append((f"KetQua_KLTN/code/{p.name}", p)); n += 1
        for extra in ("requirements.txt", "README.md"):
            if (code_dir / extra).exists():
                entries.append((f"KetQua_KLTN/code/{extra}", code_dir / extra)); n += 1
        if n == 0:
            missing.append(f"code <- {code_dir}")

    for ex in a.extra:
        p = Path(ex)
        if p.is_file():
            entries.append((f"KetQua_KLTN/tai_lieu/{p.name}", p))
        else:
            missing.append(f"--extra khong tim thay file: {ex}")

    # loc: file qua lon, file nghi co khoa
    kept, skipped = [], []
    for arc, p in entries:
        if p.stat().st_size > a.max_mb * 1e6:
            skipped.append((p, f"qua lon ({p.stat().st_size / 1e6:.0f} MB)"))
            continue
        sec = None if a.allow_secrets else has_secret(p)
        if sec:
            skipped.append((p, f"NGHI CO KHOA/MAT KHAU: ...{sec}..."))
            continue
        kept.append((arc, p))

    total = sum(p.stat().st_size for _, p in kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = ["file\tkich_thuoc_byte\tsha256"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, p in kept:
            z.write(p, arc)
            manifest.append(f"{arc}\t{p.stat().st_size}\t{sha256(p)}")
        try:
            fr = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=90).stdout
        except Exception:
            fr = "(khong lay duoc pip freeze)"
        z.writestr("KetQua_KLTN/requirements_freeze.txt", fr)
        readme = f"""KET QUA KHOA LUAN - Danh gia va tang cuong do ben vung cua mo hinh phat hien lua dao Ethereum
Dong goi luc: {datetime.now():%Y-%m-%d %H:%M} | Python {platform.python_version()} | {platform.system()}

CAC THU MUC
  code/              ma nguon cac stage (stage5 huan luyen, stage6 tan cong, stage7 phong thu, stage9 dung graph K=2,
                     convert_d2/stage8_eval_d2 cho D2) va cac script tong hop/phan tich.
  stage5_baselines/  hieu nang sach 3 mo hinh cu (GraphSAGE, GATv2, PEAE-style cu) x 3 seed; checkpoint trong models/.
  stage5_peaegnnv2/  mo hinh chinh moi peaegnnv2 (3 seed, batch 16): results_peaegnnv2.json, checkpoint trong models/.
  tai_lieu/          nhat ky thuc nghiem (.md), co so thuc te 7 tan cong (.docx) va cac file them bang --extra.
  stage6_attacks/    ket qua tan cong tren mo hinh khong phong thu. results_<mohinh>_seed<seed>.json = 6 tan cong goc;
                     hau to _adaptive_lv(_big), _bigbudget, _ablation, _newtry = cac thi nghiem bo sung/tham do.
                     stage6_summary*.md/.csv = bang trung binh +- do lech chuan qua seed.
                     results_*_mincost_*_full.json = CHI PHI TOI THIEU de vuot detector (7 tan cong x 3 mo hinh x 3 seed);
                     diag_feature_*_seed*.json = do nhay dac trung (3 mo hinh x 3 seed); mincost_summary.md, diag_summary.md = bang gop.
  stage7_defense/    train_*.json (qua trinh huan luyen), eval_seed*.json (danh gia phong thu: sach + da biet + chua thay),
                     eval_probs_*.npz (xac suat tung mau), stage7_summary/auc_analysis/stats *.md.
  stage8_D2/         danh gia truc tiep (khong huan luyen lai) cac mo hinh 2026 tren dataset XBlock D2.

DOC KET QUA NHANH
  0) tai_lieu/NhatKy_ThucNghiem_Stage5-8.md    -> TOM TAT TOAN BO (doc truoc); muc 3.5, 3.6, 3.8 la cau tra loi chinh
  1) stage6_attacks/stage6_summary.md          -> ASR cua 6 tan cong goc (mo hinh cu)
  2) stage6_attacks/stage6_summary_adaptive_lv*.md -> ADAPTIVE-STC-LV (tan cong manh nhat), ngan sach 1..500
  3) stage7_defense/stage7_summary*.md         -> phong thu: sach + ASR
  4) stage7_defense/stage7_auc_analysis*.md    -> do ben khong phu thuoc nguong (so do cong bang nhat)
  5) stage7_defense/stage7_stats*.md           -> khoang tin cay ghep cap + phan tich ty le phishing thap
  6) stage8_D2/stage8_zeroshot_d2.json         -> chuyen truc tiep sang D2

KHONG CO TRONG GOI: du lieu giao dich tho, ego_graph_*.pkl, adv_pool_train.pkl (~1 GB), file thu nghiem 1 epoch.
Cac thu nay tai tao duoc bang code (stage9 -> stage5 -> stage6 -> stage7 make_pool/train/eval).

GIOI HAN CAN GHI NHO (khi trich dan ket qua)
  - Mo hinh "PEAE-style" la ban xap xi PEAE-GNN, khong phai ban tai lap goc.
  - Chi 3 seed; do lech chuan chi phan anh khoi tao mo hinh khac nhau. Bootstrap chi the hien do bat dinh do chon mau test.
  - Cac tan cong la CO DINH, khong thich ung theo mo hinh phong thu. Chua kiem tra so du tai khoan va chi phi gas.
  - ASR chi tinh tren cac vi phishing ma mo hinh da phat hien dung luc sach.
  - D2: khong co gas; mat du lieu bac 2 khong deu giua hai lop; khac thoi ky; chi danh gia truc tiep (muc A).
  - Mo hinh ra_peae seed 44 huan luyen kem (dung som o epoch 17); da bao cao ca 3 seed.
"""
        z.writestr("KetQua_KLTN/README.txt", readme)
        z.writestr("KetQua_KLTN/MANIFEST.txt", "\n".join(manifest))

    print(f"\nDa tao: {out}  ({out.stat().st_size / 1e6:.1f} MB nen; {total / 1e6:.1f} MB goc; {len(kept)} file)")
    by = {}
    for arc, _ in kept:
        by[arc.split("/")[1]] = by.get(arc.split("/")[1], 0) + 1
    for k, v in sorted(by.items()):
        print(f"  {k:<18s} {v} file")
    if skipped:
        print("\nBO QUA:")
        for p, why in skipped:
            print(f"  - {p}  [{why}]")
    if missing:
        print("\nKHONG TIM THAY (kiem tra duong dan --root; co the chua chay buoc do):")
        for m in missing:
            print(f"  - {m}")
    if out.stat().st_size > 24e6:
        print("\nCANH BAO: file > 24 MB, de bi email tu choi. Thu --no_models hoac gui qua Drive.")
    print("\nGui file zip nay cho ban. Neu ban muon chay lai thi can them du lieu tho + pkl (xem README).")


if __name__ == "__main__":
    main()
