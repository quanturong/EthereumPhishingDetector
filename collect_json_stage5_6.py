"""
Gom file JSON ket qua Stage 5 va Stage 6 de GUI DI.

KHUYEN DUNG de gui:  python collect_json_stage5_6.py --final_only --with_summaries
  --final_only : CHI lay ket qua cua mo hinh hien dung (peae_paper, graphsage, gatv2); BO ket qua cu:
                 mo hinh PEAE-style cu (results_peaegnn_*), file thu (results_graphsage.json cua Stage 5, _try, smoke),
                 manh con sot (_shKofN), va moi bang tom tat stage6_summary* cu. Chi giu mincost_summary.md, diag_summary.md.
  Giu lai: Stage 5: results_all.json, results_peae_paper.json, stage5_rocauc*.json, diag_peae_features.json
           Stage 6: results_*_mincost_*_full.json (chi phi toi thieu), diag_feature_*_seed*.json (do nhay dac trung),
                    results_<mo hinh>_seed<seed>_adaptive_lv_big.json (ADAPTIVE ngan sach lon),
                    results_peaegnnv2_seed<seed>.json (6 tan cong, ngan sach nho)

Moi file trong INDEX.md co NGAY SUA CUOI; dung --since YYYY-MM-DD de danh dau file cu hon ngay do.
Khong gom checkpoint .pt va thu muc logs. Luon sao chep (khong xoa/sua file goc).

Chay:
  python collect_json_stage5_6.py --final_only --with_summaries          # de gui
  python collect_json_stage5_6.py                                        # gom tat ca (khong loc)
  python collect_json_stage5_6.py --final_only --since 2026-10-07        # danh dau file sua truoc 07/10/2026
"""
import argparse
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXCLUDE = ("_try", "smoke")
SHARD_RE = re.compile(r"_sh\d+of\d+(\.json)$")
FINAL5 = {"results_all.json", "results_peae_paper.json", "stage5_rocauc.json", "stage5_rocauc_merged.json", "diag_peae_features.json"}
FINAL6_RE = re.compile(r"^(results|diag_feature)_(peae_paper|graphsage|gatv2)_seed\d+(_mincost_\w+_full|_adaptive_lv_big)?\.json$")
FINAL6_SUMMARIES = {"mincost_summary.md", "diag_summary.md"}


def is_stale_shard(p: Path) -> bool:
    return bool(SHARD_RE.search(p.name)) and (p.parent / SHARD_RE.sub(r"\1", p.name)).exists()


def classify(name: str) -> str:
    if "_mincost_" in name:
        return "chi phi toi thieu (mincost)"
    if name.startswith("diag_feature_"):
        return "do nhay dac trung (diag)"
    if "_adaptive_lv_big" in name:
        return "ADAPTIVE ngan sach lon (adaptive_lv_big)"
    if name.startswith("results_"):
        return "ASR cac tan cong, ngan sach nho / khac"
    return "khac"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\NCKH\Final")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_zip", action="store_true")
    ap.add_argument("--final_only", action="store_true", help="chi lay ket qua hien dung, bo ket qua cu (khuyen dung de gui)")
    ap.add_argument("--with_summaries", action="store_true", help="them bang tom tat .md/.csv (final_only: chi mincost_summary, diag_summary)")
    ap.add_argument("--include_try", action="store_true")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD: danh dau file sua truoc ngay nay")
    ap.add_argument("--expected_mincost", type=int, default=45)
    ap.add_argument("--expected_diag", type=int, default=9)
    ap.add_argument("--expected_big", type=int, default=9)
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out) if a.out else root / ("json_stage5_6_final" if a.final_only else "json_stage5_6")
    s6 = root / "ethphishgraph2026_stage6_attacks_k2"
    s5_dirs = sorted(p for p in (root / "results").glob("stage5*") if p.is_dir())
    since = datetime.strptime(a.since, "%Y-%m-%d") if a.since else None
    if out.exists():
        shutil.rmtree(out)
    (out / "stage5").mkdir(parents=True)
    (out / "stage6").mkdir(parents=True)

    skipped_shards, dropped = [], []

    def keep(p, stage):
        if is_stale_shard(p):
            skipped_shards.append(p.name); return False
        if not a.include_try and any(x in p.name for x in EXCLUDE):
            return False
        if a.final_only:
            ok = (p.name in FINAL5) if stage == 5 else (bool(FINAL6_RE.match(p.name)) or p.name in FINAL6_SUMMARIES)
            if not ok:
                dropped.append(f"stage{stage}/{p.name}"); return False
        return True

    copied = []   # (stage, dst, size, name, mtime)
    for d in s5_dirs:
        for p in sorted(d.glob("*.json")):
            if keep(p, 5):
                dst = out / "stage5" / d.name / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                copied.append((5, dst, p.stat().st_size, p.name, datetime.fromtimestamp(p.stat().st_mtime)))
    extra5 = root / "results" / "stage5_rocauc.json"          # ROC-AUC tinh lai tu checkpoint (eval_stage5_rocauc.py)
    if extra5.exists() and keep(extra5, 5):
        dst = out / "stage5" / extra5.name
        shutil.copy2(extra5, dst)
        copied.append((5, dst, extra5.stat().st_size, extra5.name, datetime.fromtimestamp(extra5.stat().st_mtime)))
    if s6.exists():
        for pat in ["*.json"] + (["*.md", "*.csv"] if a.with_summaries else []):
            for p in sorted(s6.glob(pat)):
                if keep(p, 6):
                    dst = out / "stage6" / p.name
                    shutil.copy2(p, dst)
                    copied.append((6, dst, p.stat().st_size, p.name, datetime.fromtimestamp(p.stat().st_mtime)))
    else:
        print(f"KHONG TIM THAY thu muc Stage 6: {s6}")
    if not s5_dirs:
        print(f"KHONG TIM THAY thu muc Stage 5 trong {root / 'results'} (stage5*)")

    broken = []
    for st, dst, size, name, mt in copied:
        if dst.suffix == ".json":
            try:
                json.loads(dst.read_text(encoding="utf-8"))
            except Exception as e:
                broken.append((name, str(e)[:80]))

    groups = {}
    for st, dst, size, name, mt in copied:
        if dst.suffix == ".json":
            key = "Stage 5: ket qua huan luyen" if st == 5 else "Stage 6: " + classify(name)
            groups.setdefault(key, []).append((name, size, mt))
    cnt = lambda k: len(groups.get("Stage 6: " + k, []))
    old_files = [n for g in groups.values() for n, s, mt in g if since and mt < since]

    md = [f"# Chi muc JSON Stage 5 va Stage 6 ({datetime.now():%Y-%m-%d %H:%M}) - che do: {'CHI KET QUA HIEN DUNG (final_only)' if a.final_only else 'TAT CA (khong loc)'}", "",
          f"Nguon: `{root}`. Tong: {sum(len(v) for v in groups.values())} file JSON, {sum(s for v in groups.values() for _, s, _ in v) / 1e6:.2f} MB.", ""]
    for g, items in sorted(groups.items()):
        md += [f"## {g} ({len(items)} file)", ""]
        for n, s, mt in items:
            flag = "  <-- CU HON --since" if since and mt < since else ""
            warn = ("  <-- KIEM TRA: co the la ket qua chay thu" if g.startswith("Stage 5") and n not in FINAL5 else "")
            md.append(f"- `{n}` ({s / 1e3:.1f} kB, sua {mt:%Y-%m-%d %H:%M}){flag}{warn}")
        md.append("")
    md += ["## Kiem tra", ""]
    md.append(f"- JSON doc duoc: {'TAT CA' if not broken else 'CO LOI: ' + ', '.join(n for n, _ in broken)}")
    for lab, got, exp in (("chi phi toi thieu", cnt("chi phi toi thieu (mincost)"), a.expected_mincost),
                          ("do nhay dac trung", cnt("do nhay dac trung (diag)"), a.expected_diag),
                          ("ADAPTIVE ngan sach lon", cnt("ADAPTIVE ngan sach lon (adaptive_lv_big)"), a.expected_big)):
        md.append(f"- {lab}: {got} file (du kien {exp})" + ("" if got == exp or not a.final_only and lab == "ADAPTIVE ngan sach lon" else "  <-- KHAC DU KIEN"))
    if skipped_shards:
        md.append(f"- Bo {len(skipped_shards)} manh con sot (da co file chinh): " + ", ".join(sorted(skipped_shards)))
    if a.final_only:
        md.append(f"- BO {len(dropped)} file khong thuoc ket qua hien dung (mo hinh cu, file thu, bang tom tat cu...): " + (", ".join(sorted(dropped)) if dropped else "khong co"))
    if since:
        md.append(f"- File cu hon {a.since}: " + (", ".join(old_files) if old_files else "khong co"))
    md.append("- Khong gom: checkpoint .pt, thu muc logs" + ("" if not a.include_try else "; DA gom file thu"))
    (out / "INDEX.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    if not a.no_zip:
        z = out.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(out.name) / p.relative_to(out))
        print(f"\nDa tao: {z}  ({z.stat().st_size / 1e6:.2f} MB nen)")
    print(f"Thu muc: {out}")
    if broken:
        print("\nCANH BAO file JSON loi:", broken)


if __name__ == "__main__":
    main()
