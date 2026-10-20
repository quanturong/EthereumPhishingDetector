"""
Phep thu DO NHAY DAC TRUNG cho peae_paper (thay the diag_feature_sensitivity.py, von dung cot dac trung cu).

Tra loi "yeu to nao anh huong nhat": voi cac vi phishing mo hinh bat DUNG luc sach, thay mot nhom dac trung (trong khong
gian dac trung CUA BAI, da Min-Max) roi do ty le vi bi doi nhan (P(phishing) < 0.5). KHONG phai tan cong tren chuoi
(khong tao giao dich); chi la phep chan doan. Dung tap test trong pkl (khong can du lieu tho).

Nhom can thiep tren NODE MUC TIEU (thay bang gia tri DIEN HINH = trung vi cua cac node muc tieu NORMAL o tap train):
  Din/Dout | AFTO (thong ke gia tri ra) | AFTI (thong ke gia tri vao) | MT (ty le max/sum) | RT (cuong do tuong tac)
  | AFTS (khoang thoi gian) | tat ca 24 chieu.      Doi chung: thay bang trung vi cua node muc tieu PHISHING (train).
Can thiep tren HANG XOM (chi co Din/Dout theo bai): thay (Din, Dout) cua hang xom bac 1 / bac 2 / ca hai bang mau thuc
boc tu hang xom cua cac do thi NORMAL (train); doi chung: boc tu hang xom cua do thi PHISHING (train).
'HIEU' = ty le lat (mau normal) - ty le lat (mau phishing): phan dac hieu cua lop.

Chay:
  python diag_peae_features.py --max_test 100       # thu nhanh
  python diag_peae_features.py                       # day du, 3 seed
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

GROUPS = {   # ten nhom -> cac cot (theo thu tu dac trung cua bai)
    "Din, Dout (bac vao/ra)": [0, 1],
    "AFTO (thong ke gia tri ra)": list(range(2, 8)),
    "AFTI (thong ke gia tri vao)": list(range(8, 14)),
    "MT (ty le max/sum)": [14, 15],
    "RT (cuong do tuong tac)": [16, 17],
    "AFTS (khoang thoi gian)": list(range(18, 24)),
    "TAT CA 24 dac trung cua node muc tieu": list(range(24)),
}


def hop_masks(g):
    n, t = g["node_features"].shape[0], int(g["target_idx"])
    src, dst = g["edge_index"]
    h1 = np.zeros(n, bool)
    h1[dst[src == t]] = True
    h1[src[dst == t]] = True
    h1[t] = False
    h2 = ~h1
    h2[t] = False
    return h1, h2


def build_refs(train_raw, stats, to_pyg):
    """Trung vi dac trung node muc tieu + bo (Din, Dout) cua hang xom bac 1/2, theo nhan, chi tren TRAIN."""
    tgt = {0: [], 1: []}
    pool = {(h, y): [] for h in (1, 2) for y in (0, 1)}
    for g in train_raw:
        x = to_pyg(g, stats).x.numpy()
        y, t = int(g["label"]), int(g["target_idx"])
        tgt[y].append(x[t])
        h1, h2 = hop_masks(g)
        if h1.any():
            pool[(1, y)].append(x[h1][:, :2])
        if h2.any():
            pool[(2, y)].append(x[h2][:, :2])
    med = {y: np.median(np.stack(v), axis=0) for y, v in tgt.items()}
    pools = {k: (np.concatenate(v) if v else np.zeros((1, 2))) for k, v in pool.items()}
    return med, pools


def interventions(x, t, h1, h2, med, pools, rng):
    """Tra ve dict ten -> x moi (ban sao)."""
    out = {}
    for name, cols in GROUPS.items():
        x2 = x.copy(); x2[t, cols] = med[0][cols]
        out[name] = x2
    xc = x.copy(); xc[t] = med[1]
    out["[doi chung] thay ca node muc tieu bang trung vi PHISHING"] = xc
    for lab, hops in (("bac 1", (1,)), ("bac 2", (2,)), ("bac 1+2", (1, 2))):
        for cls, nm in ((0, "mau thuc tu NORMAL"), (1, "[doi chung] mau thuc tu PHISHING")):
            x2 = x.copy()
            for h in hops:
                m = h1 if h == 1 else h2
                if m.any():
                    p = pools[(h, cls)]
                    x2[m, :2] = p[rng.integers(0, len(p), int(m.sum()))]
            out[f"hang xom {lab} -> {nm}"] = x2
    return out


def predict_x(model, d, xs, device, bs=16):
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    outs = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            ds = []
            for x in xs[i:i + bs]:
                c = d.clone()
                c.x = torch.from_numpy(x).float()
                ds.append(c)
            b = Batch.from_data_list(ds).to(device)
            outs.append(F.softmax(model(b), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=r"D:\NCKH\Final\ethphishgraph2026_stage9_k2\ego_graph_k2_dataset.pkl")
    ap.add_argument("--peae_dir", default=r"D:\NCKH\Final\results\stage5_peae")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--max_ok", type=int, default=0, help="chi dung toi da N vi phishing bat dung moi seed (boc ngau nhien); 0 = tat ca. Nhanh hon nhung nhieu nhieu hon")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from stage5_train_baselines import MODELS, to_pyg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pdir = Path(a.peae_dir)
    stats = torch.load(pdir / "feat_stats.pt", map_location="cpu", weights_only=True)
    assert "peae_min" in stats, "feat_stats.pt khong phai cua peae_paper"
    print(f"Device: {device} | nap {a.pkl} ...")
    with open(a.pkl, "rb") as f:
        raw = pickle.load(f)
    train_raw, test_raw = raw["train"], (raw["test"][: a.max_test] if a.max_test else raw["test"])
    del raw
    gc.collect()
    print("Dung tham chieu tu tap train ...")
    med, pools = build_refs(train_raw, stats, to_pyg)
    del train_raw
    gc.collect()
    test_d = [to_pyg(g, stats) for g in test_raw]
    masks = [hop_masks(g) for g in test_raw]
    print(f"Tap test: {len(test_d)} graph")

    per_seed = {}
    for s in a.seeds:
        ck = pdir / f"model_peae_paper_seed{s}.pt"
        if not ck.exists():
            print(f"  [thieu] {ck}")
            continue
        net = MODELS["peae_paper"](in_dim=24).to(device)
        net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        net.eval()
        rng = np.random.default_rng(1000 + s)
        # sach: tim vi phishing bat dung
        clean = []
        from torch_geometric.data import Batch
        import torch.nn.functional as F
        with torch.no_grad():
            for i in range(0, len(test_d), 16):
                b = Batch.from_data_list(test_d[i:i + 16]).to(device)
                clean.append(F.softmax(net(b), dim=1)[:, 1].cpu().numpy())
        clean = np.concatenate(clean)
        ys = np.array([d.y.item() for d in test_d])
        ok = np.where((ys == 1) & (clean >= 0.5))[0]
        print(f"seed {s}: phishing bat dung luc sach = {len(ok)} / {int((ys == 1).sum())}")
        if a.max_ok and len(ok) > a.max_ok:
            ok = np.sort(rng.choice(ok, a.max_ok, replace=False))
            print(f"  (chi dung {len(ok)} vi, --max_ok)")
        acc = {}
        for i in tqdm(ok, desc=f"seed {s}", unit="vi"):
            d, t = test_d[i], int(test_d[i].target_mask.nonzero()[0])
            iv = interventions(d.x.numpy(), t, masks[i][0], masks[i][1], med, pools, rng)
            ps = predict_x(net, d, list(iv.values()), device)
            for (name, _), p in zip(iv.items(), ps):
                acc.setdefault(name, []).append((p < 0.5, p, clean[i]))
        per_seed[s] = {"n_correct_clean": int(len(ok)), "interventions": {
            n: {"flip_rate": float(np.mean([f for f, _, _ in v])), "mean_p": float(np.mean([p for _, p, _ in v])),
                "delta_p": float(np.mean([p - c for _, p, c in v]))} for n, v in acc.items()}}

    names = list(next(iter(per_seed.values()))["interventions"]) if per_seed else []
    agg = {n: (float(np.mean([r["interventions"][n]["flip_rate"] for r in per_seed.values()])),
               float(np.std([r["interventions"][n]["flip_rate"] for r in per_seed.values()]))) for n in names}
    print(f"\n{'=' * 96}\n Do nhay dac trung peae_paper: % vi phishing bi lat (trung binh ± do lech chuan qua {len(per_seed)} seed)\n{'=' * 96}")
    for n in names:
        print(f"  {n:<62s} {100 * agg[n][0]:6.1f} ± {100 * agg[n][1]:.1f}")
    print("\n  HIEU = lat(mau NORMAL) - lat(mau PHISHING, doi chung), diem %:")
    for lab in ("bac 1", "bac 2", "bac 1+2"):
        d = [100 * (r["interventions"][f"hang xom {lab} -> mau thuc tu NORMAL"]["flip_rate"]
                    - r["interventions"][f"hang xom {lab} -> [doi chung] mau thuc tu PHISHING"]["flip_rate"]) for r in per_seed.values()]
        print(f"    hang xom {lab:<8s} {np.mean(d):+6.1f} ± {np.std(d):.1f}")
    out = Path(a.out) if a.out else pdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag_peae_features.json").write_text(json.dumps({"per_seed": per_seed, "agg": agg}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out / 'diag_peae_features.json'}")


if __name__ == "__main__":
    main()