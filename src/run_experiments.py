"""
Multi-seed, longer-training experiments for the LightGCN reproduction.

Put this file in src/ next to train.py, then run:

    python src/run_experiments.py --epochs 100 --seeds 2020 2021 2022

Extra options:
    --vary_split          use a different train/test split for each seed
    --lr, --weight_decay  optimiser settings (all models)
    --bpr_lr, --bpr_wd    override the settings for BPR-MF only
    --out DIR             output folder (use a new one for each experiment)

It reuses BPRDataset and evaluate() from train.py, and records the full
training curve (loss, Recall@20, NDCG@20) for every model and seed.

Outputs (in --out, default: results/):
    curves.csv    model, seed, epoch, loss, recall, ndcg
    summary.csv   final-epoch mean and std over seeds, per model
    curves.png    mean Recall@20 vs epoch per model (shaded = +/- 1 std)

Note: the train/test split comes from load_movielens() and is NOT changed
between seeds. Seeds vary embedding initialisation and negative sampling.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import load_movielens, build_norm_adj
from models import BPRMF, LightGCN, bpr_loss
from train import BPRDataset, evaluate


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_one(kind, n_layers, seed, train, test, n_users, n_items, norm_adj,
            epochs, eval_every, batch_size=2048, lr=1e-3, weight_decay=1e-5):
    """Train one model with one seed. Returns a list of curve rows."""
    set_seed(seed)
    dataset = BPRDataset(train, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if kind == "bprmf":
        model = BPRMF(n_users, n_items, emb_dim=64)
        is_lgcn = False
    else:
        model = LightGCN(n_users, n_items, emb_dim=64, n_layers=n_layers, norm_adj=norm_adj)
        is_lgcn = True
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for u, pos, neg in loader:
            optimizer.zero_grad()
            if is_lgcn:
                users_emb, items_emb = model()
                u_emb, p_emb, n_emb = users_emb[u], items_emb[pos], items_emb[neg]
                pos_scores = (u_emb * p_emb).sum(dim=1)
                neg_scores = (u_emb * n_emb).sum(dim=1)
            else:
                pos_scores = model(u, pos)
                neg_scores = model(u, neg)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % eval_every == 0 or epoch == epochs:
            r, n = evaluate(model, train, test, n_items, k=20, is_lightgcn=is_lgcn)
            curve.append((epoch, total_loss / len(loader), float(r), float(n)))
            print(f"  [{kind}{'-%dL' % n_layers if is_lgcn else ''} seed={seed}] "
                  f"epoch {epoch}: loss={curve[-1][1]:.4f} "
                  f"Recall@20={r:.4f} NDCG@20={n:.4f}", flush=True)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2020, 2021, 2022])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--vary_split", action="store_true")
    ap.add_argument("--split_seed", type=int, default=2020)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--bpr_lr", type=float, default=None)
    ap.add_argument("--bpr_wd", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = {}

    def get_data(seed):
        split_seed = seed if args.vary_split else args.split_seed
        if split_seed not in cache:
            print(f"Loading data (split seed {split_seed})...")
            nu, ni, tr, te = load_movielens(seed=split_seed)
            cache[split_seed] = (nu, ni, tr, te, build_norm_adj(nu, ni, tr))
            print(f"users={nu}, items={ni}")
        return cache[split_seed]

    configs = [("BPR-MF", "bprmf", 0)] + [(f"LightGCN-{l}L", "lightgcn", l) for l in args.layers]
    rows = []  # (model, seed, epoch, loss, recall, ndcg)
    for name, kind, nl in configs:
        for seed in args.seeds:
            print(f"\n=== {name}, seed {seed} ===", flush=True)
            n_users, n_items, train, test, norm_adj = get_data(seed)
            lr, wd = args.lr, args.weight_decay
            if kind == "bprmf":
                lr = args.bpr_lr if args.bpr_lr is not None else lr
                wd = args.bpr_wd if args.bpr_wd is not None else wd
            curve = run_one(kind, nl, seed, train, test, n_users, n_items,
                            norm_adj, args.epochs, args.eval_every,
                            lr=lr, weight_decay=wd)
            rows += [(name, seed, *c) for c in curve]

    # raw curves
    with open(os.path.join(args.out, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "epoch", "loss", "recall", "ndcg"])
        w.writerows(rows)

    # summary at the final epoch (no picking the best epoch on the test set)
    names = [c[0] for c in configs]
    final = {n: {"recall": [], "ndcg": []} for n in names}
    per_seed = {n: {} for n in names}
    for m, s, e, _, r, nd in rows:
        if e == args.epochs:
            final[m]["recall"].append(r)
            final[m]["ndcg"].append(nd)
            per_seed[m][s] = r

    print(f"\n=== FINAL RESULTS (epoch {args.epochs}, {len(args.seeds)} seeds, mean +/- std) ===")
    print(f"{'Model':<16} {'Recall@20':<20} {'NDCG@20':<20}")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "recall_mean", "recall_std", "ndcg_mean", "ndcg_std", "n_seeds"])
        for n in names:
            r, nd = np.array(final[n]["recall"]), np.array(final[n]["ndcg"])
            rs, ns = (r.std(ddof=1), nd.std(ddof=1)) if len(r) > 1 else (0.0, 0.0)
            print(f"{n:<16} {r.mean():.4f} +/- {rs:.4f}    {nd.mean():.4f} +/- {ns:.4f}")
            w.writerow([n, r.mean(), rs, nd.mean(), ns, len(r)])

    # how consistent is the LightGCN-vs-BPR-MF gap across seeds?
    for n in names[1:]:
        wins = sum(per_seed[n][s] > per_seed["BPR-MF"][s] for s in args.seeds)
        print(f"{n} beats BPR-MF (Recall@20) in {wins}/{len(args.seeds)} seeds")

    # curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for n in names:
            epochs = sorted({e for m, _, e, *_ in rows if m == n})
            M = np.array([[r for m, s, e, _, r, _ in rows if m == n and e == ep] for ep in epochs])
            mean, std = M.mean(axis=1), M.std(axis=1)
            ax.plot(epochs, mean, label=n)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall@20")
        ax.legend(frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "curves.png"), dpi=200)
        print(f"Saved plot to {os.path.join(args.out, 'curves.png')}")
    except ImportError:
        print("matplotlib not installed; skipped plot.")


if __name__ == "__main__":
    main()