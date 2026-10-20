"""
Tinh lai ROC-AUC cho cac mo hinh Stage 5 TU CHECKPOINT DA LUU (KHONG huan luyen lai).

Vi sao duoc: file JSON cua Stage 5 khong luu ROC-AUC, nhung checkpoint (model_*_seed*.pt) va tap test van con.
Chi can cho tung mo hinh du doan xac suat tren tap test (1.408 target) roi tinh ROC-AUC.

Kiem tra tinh dung dan: script tinh lai CA macro-F1 va PR-AUC va so voi so da luu trong results_*.json;
neu lech nhieu (> 0.002) thi co gi do khac (khac tap test, khac chuan hoa, khac checkpoint) -> khong tin ROC-AUC.

Chay:
  python eval_stage5_rocauc.py --max_test 100        # thu nhanh (so sanh voi JSON se lech vi chi 100 mau)
  python eval_stage5_rocauc.py                       # day du: graphsage, gatv2, peaegnnv2 x 3 seed
  python eval_stage5_rocauc.py --models peaegnnv2 --seeds 42

Luu y bo nho: file pkl ~4 GB, nap vao RAM can nhieu GB (script giai phong train/val ngay sau khi nap).
"""
import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_PKL = r"D:\NCKH\Final\ethphishgraph2026_stage9_k2\ego_graph_k2_dataset.pkl"
DEFAULT_STAGE5 = r"D:\NCKH\Final\results\stage5_k2"
DEFAULT_V2 = r"D:\NCKH\Final\results\stage5_k2_7_10"
DEFAULT_PEAE = r"D:\NCKH\Final\results\stage5_peae"


def predict_probs(model, graphs, device, bs):
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=bs):
            batch = batch.to(device)
            probs.append(F.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


def metrics(y, p):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    pred = (p >= 0.5).astype(int)          # tuong duong argmax cua softmax 2 lop
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "rec_phishing": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))}


def stored(stage5_dir, v2_dir, peae_dir, model, seed_idx):
    """So da luu trong results_*.json (de doi chieu); None neu khong co."""
    for d, name in ((v2_dir, f"results_{model}.json"), (peae_dir, f"results_{model}.json"), (stage5_dir, "results_all.json"), (stage5_dir, f"results_{model}.json")):
        f = Path(d) / name
        if f.exists():
            try:
                r = json.loads(f.read_text(encoding="utf-8"))[model]["per_seed"][seed_idx]["test"]
                return {"macro_f1": r["macro_f1"], "pr_auc": r["pr_auc"], "src": f.name}
            except Exception:
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--stage5_dir", default=DEFAULT_STAGE5)
    ap.add_argument("--v2_dir", default=DEFAULT_V2)
    ap.add_argument("--peae_dir", default=DEFAULT_PEAE, help="thu muc cua peae_paper (co feat_stats.pt rieng)")
    ap.add_argument("--models", nargs="+", default=["peaegnnv2", "graphsage", "gatv2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=0.002)
    a = ap.parse_args()

    import torch
    from stage5_train_baselines import MODELS, to_pyg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.out) if a.out else Path(a.stage5_dir).parent
    stats = torch.load(Path(a.stage5_dir) / "feat_stats.pt", map_location="cpu", weights_only=True)
    print(f"Device: {device} | nap {a.pkl} ...")
    with open(a.pkl, "rb") as f:
        raw = pickle.load(f)
    test_raw = raw["test"][: a.max_test] if a.max_test else raw["test"]
    raw["train"] = raw["val"] = None
    gc.collect()
    graphs = [to_pyg(g, stats) for g in test_raw] if any(m != "peae_paper" for m in a.models) else []
    graphs_peae = []
    if "peae_paper" in a.models:                 # mo hinh nay co bo dac trung va thong ke rieng
        st_p = torch.load(Path(a.peae_dir) / "feat_stats.pt", map_location="cpu", weights_only=True)
        graphs_peae = [to_pyg(g, st_p) for g in test_raw]
    del raw, test_raw
    gc.collect()
    in_dim = int(stats["node_mean"].numel())
    ys = np.array([g.y.item() for g in (graphs or graphs_peae)])
    print(f"Tap test: {len(graphs or graphs_peae)} graph, ty le phishing = {ys.mean():.1%}")

    results, probs_store, worst_diff = {}, {}, 0.0
    for m in a.models:
        results[m] = []
        ck_dir = Path(a.v2_dir) if m == "peaegnnv2" else (Path(a.peae_dir) if m == "peae_paper" else Path(a.stage5_dir))
        for i, s in enumerate(a.seeds):
            ck = ck_dir / f"model_{m}_seed{s}.pt"
            if not ck.exists():
                print(f"  [thieu] {ck}")
                continue
            net = MODELS[m](in_dim=24 if m == "peae_paper" else in_dim).to(device)
            net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            y, p = predict_probs(net, graphs_peae if m == "peae_paper" else graphs, device, 16 if m == "peaegnnv2" else 32)
            r = metrics(y, p)
            st = stored(a.stage5_dir, a.v2_dir, a.peae_dir, m, i)
            if st and not a.max_test:
                r["diff_macro_f1_vs_json"] = r["macro_f1"] - st["macro_f1"]
                r["diff_pr_auc_vs_json"] = r["pr_auc"] - st["pr_auc"]
                worst_diff = max(worst_diff, abs(r["diff_macro_f1_vs_json"]), abs(r["diff_pr_auc_vs_json"]))
            r["seed"] = s
            results[m].append(r)
            probs_store[f"{m}__seed{s}"] = p
            print(f"  {m:<10s} seed{s}: ROC-AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f} macroF1={r['macro_f1']:.4f}"
                  + (f" | lech so voi JSON: F1 {r['diff_macro_f1_vs_json']:+.4f}, PR-AUC {r['diff_pr_auc_vs_json']:+.4f}" if "diff_macro_f1_vs_json" in r else ""))
            del net
            gc.collect()

    print(f"\n{'=' * 78}\n ROC-AUC tren tap test (trung binh ± do lech chuan qua seed)\n{'=' * 78}")
    print(f"{'Mo hinh':<12s}{'so seed':>8s}{'ROC-AUC':>18s}{'PR-AUC':>18s}{'Macro-F1':>18s}")
    agg = {}
    for m, rs in results.items():
        if not rs:
            continue
        f = lambda k: f"{np.mean([r[k] for r in rs]):.4f} ± {np.std([r[k] for r in rs]):.4f}"
        agg[m] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in ("roc_auc", "pr_auc", "macro_f1", "rec_phishing")}
        print(f"{m:<12s}{len(rs):>8d}{f('roc_auc'):>18s}{f('pr_auc'):>18s}{f('macro_f1'):>18s}")
    if not a.max_test:
        ok = worst_diff <= a.tol
        print(f"\nDoi chieu voi JSON da luu: lech lon nhat = {worst_diff:.4f} -> " +
              ("KHOP (ROC-AUC dang tin)" if ok else f"LECH > {a.tol}: KHONG tin ROC-AUC, gui Claude xem"))
    else:
        print("\n(--max_test: chi thu nhanh, khong doi chieu voi JSON)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage5_rocauc.json").write_text(json.dumps({"per_seed": results, "agg": agg, "n_test": len(graphs or graphs_peae),
                                                           "max_abs_diff_vs_json": worst_diff}, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "stage5_test_probs.npz", labels=ys, **probs_store)
    print(f"\nSaved -> {out_dir / 'stage5_rocauc.json'}\nSaved -> {out_dir / 'stage5_test_probs.npz'}")


if __name__ == "__main__":
    main()
