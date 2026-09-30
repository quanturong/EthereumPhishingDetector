"""
Chuan bi bundle dataset de host qua HTTP.
Copy files can thiet vao 1 folder de serve.

Cach chay:
  python prepare_dataset_bundle.py --scenario B     # K=1 + attack raw txs
  python prepare_dataset_bundle.py --scenario A --k k1    # chi pkl K=1
  python prepare_dataset_bundle.py --scenario A --k k2    # chi pkl K=2
  python prepare_dataset_bundle.py --scenario C     # K=2 full (large)
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(r"E:\EthereumPhishingDetection")
DATA = BASE / "dataset" / "EthPhishGraph-2026"
OUT  = BASE / "dataset_bundle"


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        print(f"  [skip] {src} not exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  [file] {src.name} ({size_mb:.1f} MB)")
    else:
        if dst.exists():
            print(f"  [skip] {dst} already exists")
            return
        print(f"  [copy] {src} -> {dst}")
        shutil.copytree(src, dst)


def scenario_A(k: str):
    """Chi ego-graph pkl + split metadata."""
    print(f"\n=== Scenario A ({k}): baseline training only ===")
    if k == "k1":
        copy_tree(DATA / "ethphishgraph2026_stage3_dataset" / "ego_graph_dataset.pkl",
                  OUT / "ego_graph_dataset.pkl")
    else:
        copy_tree(DATA / "ethphishgraph2026_stage9_k2" / "ego_graph_k2_dataset.pkl",
                  OUT / "ego_graph_k2_dataset.pkl")
    copy_tree(DATA / "ethphishgraph2026_stage3_dataset" / "temporal_split.csv",
              OUT / "temporal_split.csv")
    copy_tree(DATA / "ethphishgraph2026_stage3_dataset" / "temporal_split_meta.json",
              OUT / "temporal_split_meta.json")


def scenario_B():
    """K=1 pkl + hop-1 raw txs (train + attack tren K=1)."""
    print(f"\n=== Scenario B: train + attack K=1 (~540 MB) ===")
    scenario_A("k1")
    copy_tree(DATA / "ethphishgraph2026_stage2_normal_crawl" / "hop1_txs",
              OUT / "phishing_hop1_txs")
    copy_tree(DATA / "ethphishgraph2026_stage3_dataset" / "normal_hop1_txs",
              OUT / "normal_hop1_txs")


def scenario_C():
    """K=2 full: pkl + hop-1 + hop-2 raw (train + attack K=2, ~26 GB)."""
    print(f"\n=== Scenario C: train + attack K=2 (~26 GB) ===")
    scenario_A("k2")
    copy_tree(DATA / "ethphishgraph2026_stage2_normal_crawl" / "hop1_txs",
              OUT / "phishing_hop1_txs")
    copy_tree(DATA / "ethphishgraph2026_stage3_dataset" / "normal_hop1_txs",
              OUT / "normal_hop1_txs")
    copy_tree(DATA / "ethphishgraph2026_stage9_k2" / "hop2_neighbor_txs",
              OUT / "hop2_neighbor_txs")
    copy_tree(DATA / "ethphishgraph2026_stage9_k2" / "per_target_neighbors.pkl",
              OUT / "per_target_neighbors.pkl")


def add_source_code():
    """Copy source scripts de nguoi tai co the chay."""
    print("\n=== Copy source scripts ===")
    (OUT / "src").mkdir(exist_ok=True)
    for f in ["stage3_pipeline.py", "stage5_train_baselines.py",
              "stage6_adversarial_attacks.py", "stage7_defense.py",
              "crawl_normal_candidates.py"]:
        src = BASE / f
        if src.exists():
            shutil.copy2(src, OUT / "src" / f)
            print(f"  [src] {f}")


def write_readme(scenario: str, k: str):
    readme = OUT / "README.md"
    total_size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    with open(readme, "w", encoding="utf-8") as fp:
        fp.write(f"""# EthPhishGraph-2026 Dataset Bundle

**Scenario**: {scenario} ({k if scenario=='A' else '-'})
**Total size**: {total_size:.1f} MB

## Contents

""")
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                rel = p.relative_to(OUT)
                size = p.stat().st_size / 1e6
                fp.write(f"- `{rel}` ({size:.2f} MB)\n")
        fp.write(f"""

## Quick start

```python
import pickle
with open('ego_graph_dataset.pkl', 'rb') as f:  # or ego_graph_k2_dataset.pkl
    dataset = pickle.load(f)
# dataset = {{"train": [...], "val": [...], "test": [...]}}
# each item: dict with keys target, target_idx, label, edge_index,
# edge_attr, node_features, addresses, n_nodes, n_edges, partition
```

See `src/` for training scripts.
""")
    print(f"\n  [readme] {readme}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["A", "B", "C"], required=True)
    p.add_argument("--k", choices=["k1", "k2"], default="k1")
    p.add_argument("--no_src", action="store_true", help="Bo qua source code")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    print(f"Output bundle: {OUT}")

    if   args.scenario == "A": scenario_A(args.k)
    elif args.scenario == "B": scenario_B()
    elif args.scenario == "C": scenario_C()

    if not args.no_src:
        add_source_code()

    write_readme(args.scenario, args.k)

    total_size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n=== Done ===\nBundle size: {total_size/1e6:.1f} MB ({total_size/1e9:.2f} GB)")
    print(f"Location: {OUT}")


if __name__ == "__main__":
    main()
