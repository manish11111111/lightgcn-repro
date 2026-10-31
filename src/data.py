import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_movielens(path="data/ml-latest-small/ratings.csv",
                   threshold=4.0, test_ratio=0.2, seed=2020):
    """Rating >= threshold lai positive interaction maanera
    har user ko interactions train/test ma split garchha."""
    df = pd.read_csv(path)
    df = df[df["rating"] >= threshold].copy()

    user_map = {u: i for i, u in enumerate(sorted(df["userId"].unique()))}
    item_map = {m: i for i, m in enumerate(sorted(df["movieId"].unique()))}
    df["u"] = df["userId"].map(user_map)
    df["i"] = df["movieId"].map(item_map)

    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for u, g in df.groupby("u"):
        items = g["i"].values.copy()
        rng.shuffle(items)
        if len(items) < 2:
            train[u], test[u] = items, np.array([], dtype=int)
            continue
        n_test = max(1, int(round(test_ratio * len(items))))
        test[u], train[u] = items[:n_test], items[n_test:]

    return len(user_map), len(item_map), train, test


def build_norm_adj(n_users, n_items, train):
    """LightGCN ko normalized adjacency: D^-1/2 A D^-1/2,
    jaha A = [[0, R], [R^T, 0]]."""
    rows, cols = [], []
    for u, items in train.items():
        rows.extend([u] * len(items))
        cols.extend(items)
    R = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(n_users, n_items), dtype=np.float32)

    zero_u = sp.csr_matrix((n_users, n_users), dtype=np.float32)
    zero_i = sp.csr_matrix((n_items, n_items), dtype=np.float32)
    A = sp.bmat([[zero_u, R], [R.T, zero_i]], format="coo")

    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = deg[nz] ** -0.5
    D = sp.diags(d_inv_sqrt)
    norm = (D @ A @ D).tocoo()

    idx = torch.tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    val = torch.tensor(norm.data, dtype=torch.float32)
    size = (n_users + n_items, n_users + n_items)
    return torch.sparse_coo_tensor(idx, val, size).coalesce()


if __name__ == "__main__":
    n_users, n_items, train, test = load_movielens()
    adj = build_norm_adj(n_users, n_items, train)
    n_train = sum(len(v) for v in train.values())
    n_test = sum(len(v) for v in test.values())
    print(f"users={n_users}, items={n_items}")
    print(f"train interactions={n_train}, test interactions={n_test}")
    print(f"adj shape={tuple(adj.shape)}, nnz={adj._nnz()}")