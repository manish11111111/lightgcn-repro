"""
Hyperparameter tuning on a VALIDATION split (never on the test set).

10% of each user's training items are held out as validation. Each config is
trained on the remaining training items and scored on validation Recall@20
at the final epoch. The test set is not touched.

    python src/tune.py --epochs 100

Then rerun the main experiment with the chosen values, e.g.:
    python src/run_experiments.py --epochs 100 --lr 0.003 --weight_decay 1e-5 \\
        --bpr_lr 0.01 --bpr_wd 1e-5 --out results_tuned
"""
import argparse
import csv
import itertools
import os

import numpy as np

from data import load_movielens, build_norm_adj
from run_experiments import run_one


def make_val_split(train, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    tr, val = {}, {}
    for u, items in train.items():
        items = np.array(items).copy()
        rng.shuffle(items)
        n_val = max(1, int(round(frac * len(items)))) if len(items) >= 5 else 0
        val[u], tr[u] = items[:n_val], items[n_val:]
    return tr, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    ap.add_argument("--out", type=str, default="results_tuning")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n_users, n_items, train, _test = load_movielens()
    tr, val = make_val_split(train)
    norm_adj = build_norm_adj(n_users, n_items, tr)

    rows = []
    for (name, kind, nl), lr, wd in itertools.product(
            [("BPR-MF", "bprmf", 0), ("LightGCN-1L", "lightgcn", 1)], args.lrs, args.wds):
        print(f"\n=== {name} lr={lr} wd={wd} ===", flush=True)
        curve = run_one(kind, nl, args.seed, tr, val, n_users, n_items, norm_adj,
                        args.epochs, eval_every=args.epochs, lr=lr, weight_decay=wd)
        _, _, rec, ndcg = curve[-1]
        rows.append((name, lr, wd, rec, ndcg))

    with open(os.path.join(args.out, "tuning.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "lr", "weight_decay", "val_recall", "val_ndcg"])
        w.writerows(rows)

    print("\n=== VALIDATION RESULTS (Recall@20, final epoch) ===")
    for name in ("BPR-MF", "LightGCN-1L"):
        sub = sorted([r for r in rows if r[0] == name], key=lambda r: -r[3])
        for r in sub:
            print(f"{name:<12} lr={r[1]:<7} wd={r[2]:<7} val Recall@20={r[3]:.4f}  NDCG@20={r[4]:.4f}")
        print(f"--> best for {name}: lr={sub[0][1]}, weight_decay={sub[0][2]}\n")


if __name__ == "__main__":
    main()