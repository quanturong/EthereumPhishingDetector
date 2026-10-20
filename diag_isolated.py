"""
Chan doan node co lap trong ego-graph K=2.
Chay (cung thu muc voi stage9_k2_upgrade.py):
    python diag_isolated.py            # quet 300 target dau
    python diag_isolated.py 1000       # quet 1000 target
Gui toan bo output cho Claude.
"""
import sys
from collections import Counter
import numpy as np
import pandas as pd
import stage9_k2_upgrade as m

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
split = pd.read_csv(m.SPLIT_CSV)
meta = __import__("json").loads(m.SPLIT_META.read_text(encoding="utf-8"))
T_end = int(split["anchor_ts"].max()) + 10 * 365 * 24 * 3600
cutoffs = {"train": meta["T1_ts"], "val": meta["T2_ts"], "test": T_end}
label_of = dict(zip(split["address"].str.lower(), split["label"].astype(int)))

kinds, shown, n_graph_iso = Counter(), 0, 0
for _, row in split.head(N).iterrows():
    addr, label = row["address"].lower(), int(row["label"])
    cutoff = cutoffs[row["partition"]]
    tgt_txs = m._load_hop1(addr, label)
    h1 = m._hop1_before_cutoff(addr, tgt_txs, cutoff)
    nmap = {}
    for n in h1:
        nt = (m._load_hop1(n, label_of[n])[:m.MAX_TX_PER_NEIGHBOR]
              if m.USE_TARGET_FILES and n in label_of else m._load_neighbor_txs(n))
        if nt:
            nmap[n] = nt
    g = m._build_k2_ego(addr, label, cutoff, h1, tgt_txs, nmap)
    if g is None:
        continue
    deg = np.bincount(g["edge_index"].ravel(), minlength=g["n_nodes"])
    iso = [g["addresses"][i] for i in np.where(deg == 0)[0]]
    if not iso:
        continue
    n_graph_iso += 1
    for a in iso:
        kind = "hop1" if a in h1 else ("target" if a == addr else "hop2")
        kinds[kind] += 1
        if shown < 8:
            shown += 1
            print(f"\n[{row['partition']}] target={addr[:12]}.. isolated={a} kind={kind} "
                  f"in_neigh_map={a in nmap}")
            hits = [(src, t) for src, lst in [("target", tgt_txs)] + list(nmap.items())
                    for t in lst
                    if a in ((t.get('from') or '').lower(), (t.get('to') or '').lower())]
            print(f"  so tx nhac toi node nay trong du lieu: {len(hits)}")
            for src, t in hits[:3]:
                print(f"   nguon={str(src)[:12]} ts={t.get('timeStamp')} (cutoff={cutoff}) "
                      f"from={(t.get('from') or '')[:12]} to={(t.get('to') or '')[:12]} "
                      f"hash={'co' if t.get('hash') else 'THIEU'}")
print(f"\nQuet {N} target: {n_graph_iso} graph co node co lap. Phan loai: {dict(kinds)}")