"""
Chay do MOC VUOT TOI THIEU (stage6 --min_cost) cho 7 tan cong x nhieu mo hinh x nhieu seed, bang 1 lenh.

Vi sao chia nhom: moi don vi ngan sach b la 1 (TOH, TIH), 2 (BPH), 3 (MWR), 4 (STC), 6 (ADAPTIVE) giao dich.
Moi nhom co luoi b rieng sao cho tat ca cung quet toi ~1.200 giao dich gia (de so sanh theo SO GIAO DICH).
Moi lan chay luu 1 file: results_<mo_hinh>_seed<seed>_mincost_<nhom>.json; da co file thi bo qua (chay lai duoc sau khi bi ngat).

Chay:
  python run_mincost_all.py --dry_run                      # chi in cac lenh, khong chay
  python run_mincost_all.py --models peaegnnv2 --max_targets 30 --seeds 42 --tag_suffix try   # thu nhanh 1 mo hinh
  python run_mincost_all.py                                # peaegnnv2 + graphsage + gatv2, seed 42
  python run_mincost_all.py --seeds 42 43 44               # du 3 seed
  python run_mincost_all.py --parallel 3                   # chay 3 lan cung luc (xem ghi chu ben duoi)
  python run_mincost_all.py --shards 4 --parallel 3        # chia moi lan chay thanh 4 manh ngan, chay 3 manh cung luc
  python run_mincost_all.py --shards 4 --only_shards 1 2   # may A chay manh 1,2; may B chay --only_shards 3 4 (roi gop file ket qua)

Chay song song (--parallel N): moi lan chay la 1 tien trinh Python rieng, ket qua in vao file log trong thu muc logs trong out_dir.
Moi tien trinh duoc gioi han so luong CPU (OMP_NUM_THREADS = so nhan // N) de khong tranh nhau. Cang nhieu tien trinh
cang ton RAM (moi tien trinh ~ vai GB); neu may bi cham/het RAM, giam N. Kiem tra so nhan:  echo $env:NUMBER_OF_PROCESSORS
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GRIDS = {   # nhom: (cac tan cong, luoi ngan sach b, tx moi don vi)
    "toh_tih":  ("TOH TIH",             [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1200]),
    "bph":      ("BPH-T2P BPH-P2T",     [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500, 600]),
    "mwr":      ("MWR",                 [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400]),
    "stc":      ("STC",                 [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300]),
    "adaptive": ("ADAPTIVE-STC-LV",     [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200]),
}
MODEL_DIRS = {
    "peaegnnv2": r"D:\NCKH\Final\results\stage5_k2_7_10",
    "graphsage": r"D:\NCKH\Final\results\stage5_k2",
    "gatv2":     r"D:\NCKH\Final\results\stage5_k2",
    "peaegnn":   r"D:\NCKH\Final\results\stage5_k2",
    "peae_paper": r"D:\NCKH\Final\results\stage5_peae",    # PEAE-GNN ban giong bai (dac trung rieng)
}
OUT_DIR = Path(r"D:\NCKH\Final\ethphishgraph2026_stage6_attacks_k2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["peaegnnv2", "graphsage", "gatv2"], choices=list(MODEL_DIRS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--groups", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    ap.add_argument("--max_targets", type=int, default=0)
    ap.add_argument("--tag_suffix", default="", help="them vao tag (vd 'try' khi chay thu, de khong lan voi ket qua that)")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="chay lai ca khi da co file ket qua")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--parallel", type=int, default=1, help="so lan chay cung luc (mac dinh 1)")
    ap.add_argument("--shards", type=int, default=1,
                    help="chia vi phishing thanh N manh; moi manh la 1 lan chay rieng, luu rieng (chay ngan, ngat/chay lai it mat cong)")
    ap.add_argument("--only_shards", nargs="+", type=int, default=None,
                    help="chi chay cac manh nay (de chia cho nhieu may: may A --only_shards 1 2, may B --only_shards 3 4)")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    jobs = []
    shard_ids = [k for k in range(1, a.shards + 1) if not a.only_shards or k in a.only_shards]
    for s in a.seeds:
        for m in a.models:
            for g in a.groups:
                atks, budgets = GRIDS[g]
                for k in shard_ids:
                    tag = g + (f"_{a.tag_suffix}" if a.tag_suffix else "") + (f"_sh{k}of{a.shards}" if a.shards > 1 else "")
                    out = Path(a.out_dir) / f"results_{m}_seed{s}_mincost_{tag}.json"
                    cmd = [sys.executable, str(here / "stage6_adversarial_attacks.py"), "--model", m, "--ckpt_seed", str(s),
                           "--min_cost", "--attacks", *atks.split(), "--budgets", *map(str, budgets),
                           "--tag", tag, "--stage5_dir", MODEL_DIRS[m]]
                    if a.shards > 1:
                        cmd += ["--shard", f"{k}/{a.shards}"]
                    if a.max_targets:
                        cmd += ["--max_targets", str(a.max_targets)]
                    jobs.append((m, s, g if a.shards == 1 else f"{g}[{k}/{a.shards}]", out, cmd))

    print(f"Tong {len(jobs)} lan chay (seed {a.seeds} x mo hinh {a.models} x nhom {a.groups}), song song = {a.parallel}.")
    todo = []
    for i, (m, s, g, out, cmd) in enumerate(jobs, 1):
        skip = out.exists() and not a.force
        print(f"[{i}/{len(jobs)}] {m} seed{s} nhom={g} {'(DA CO -> bo qua)' if skip else ''}")
        if a.dry_run:
            print("   " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        elif not skip:
            todo.append((i, m, s, g, out, cmd))
    if a.dry_run:
        return

    if a.parallel <= 1:
        for i, m, s, g, out, cmd in todo:
            print(f"\n>>> [{i}/{len(jobs)}] {m} seed{s} nhom={g}")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"   LOI (ma {r.returncode}) o {m} seed{s} {g}. Dung lai; gui nguyen van loi cho Claude.")
                sys.exit(r.returncode)
    else:
        log_dir = Path(a.out_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        threads = max(1, (os.cpu_count() or 2) // a.parallel)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        print(f"Moi tien trinh dung toi da {threads} luong CPU. Log: {log_dir}")

        def work(job):
            i, m, s, g, out, cmd = job
            safe = g.replace("[", "_sh").replace("]", "").replace("/", "of")
            log = log_dir / f"{m}_seed{s}_{safe}.log"
            t0 = time.time()
            try:
                with open(log, "w", encoding="utf-8") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            except Exception as e:           # 1 lan loi khong duoc lam sap ca dot chay
                rc = -1
                print(f"   [LOI he thong] {m} seed{s} {g}: {e}")
            return job, rc, (time.time() - t0) / 60, log

        failed = []
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            futs = [ex.submit(work, j) for j in todo]
            for fut in as_completed(futs):
                (i, m, s, g, out, cmd), rc, mins, log = fut.result()
                status = "xong" if rc == 0 else f"LOI (ma {rc}) - xem {log}"
                print(f"[{i}/{len(jobs)}] {m} seed{s} nhom={g}: {status} ({mins:.1f} phut)")
                if rc != 0:
                    failed.append((m, s, g))
        if failed:
            sys.exit(f"{len(failed)} lan chay bi loi: {failed}. Gui nguyen van log (file .log) cho Claude.")
    print("\nXong. Gop ket qua:  python summarize_mincost.py")


if __name__ == "__main__":
    main()
