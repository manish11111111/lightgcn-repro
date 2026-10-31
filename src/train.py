import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

from data import load_movielens, build_norm_adj
from models import BPRMF, LightGCN, bpr_loss


class BPRDataset(Dataset):
    def __init__(self, train, n_items):
        self.users, self.pos_items, self.neg_items = [], [], []
        for u, items in train.items():
            if len(items) == 0:
                continue
            for i in items:
                self.users.append(u)
                self.pos_items.append(i)
                self.neg_items.append(-1)  # will sample on the fly
        self.n_items = n_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]
        neg = random.randint(0, self.n_items - 1)
        return u, pos, neg


def evaluate(model, train, test, n_items, k=20, is_lightgcn=False):
    model.eval()
    with torch.no_grad():
        if is_lightgcn:
            scores = model.get_all_scores()
        else:
            users = torch.arange(model.user_emb.num_embeddings)
            scores = model.get_all_scores(users, n_items)

    recalls, ndcgs = [], []
    for u, test_items in test.items():
        if len(test_items) == 0:
            continue
        # Exclude training items
        train_items = set(train.get(u, []))
        user_scores = scores[u].clone()
        if len(train_items) > 0:
            user_scores[list(train_items)] = -float("inf")

        # Top-K
        _, topk = torch.topk(user_scores, k)
        topk = topk.cpu().numpy()
        hits = set(topk) & set(test_items)

        recall = len(hits) / len(test_items)
        recalls.append(recall)

        # NDCG
        dcg = 0.0
        for rank, item in enumerate(topk):
            if item in test_items:
                dcg += 1.0 / np.log2(rank + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(recalls), np.mean(ndcgs)


def train_bpr_mf(train, test, n_users, n_items, epochs=20, batch_size=2048, lr=1e-3):
    print("\n=== Training BPR-MF ===")
    dataset = BPRDataset(train, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BPRMF(n_users, n_items, emb_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for u, pos, neg in loader:
            optimizer.zero_grad()
            pos_scores = model(u, pos)
            neg_scores = model(u, neg)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            recall, ndcg = evaluate(model, train, test, n_items, k=20, is_lightgcn=False)
            print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, "
                  f"Recall@20={recall:.4f}, NDCG@20={ndcg:.4f}")

    recall, ndcg = evaluate(model, train, test, n_items, k=20, is_lightgcn=False)
    return model, recall, ndcg


def train_lightgcn(train, test, n_users, n_items, norm_adj, epochs=20,
                   batch_size=2048, lr=1e-3, n_layers=3):
    print(f"\n=== Training LightGCN (layers={n_layers}) ===")
    dataset = BPRDataset(train, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = LightGCN(n_users, n_items, emb_dim=64, n_layers=n_layers, norm_adj=norm_adj)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for u, pos, neg in loader:
            optimizer.zero_grad()
            users_emb, items_emb = model()
            u_emb = users_emb[u]
            pos_emb = items_emb[pos]
            neg_emb = items_emb[neg]
            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            recall, ndcg = evaluate(model, train, test, n_items, k=20, is_lightgcn=True)
            print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, "
                  f"Recall@20={recall:.4f}, NDCG@20={ndcg:.4f}")

    recall, ndcg = evaluate(model, train, test, n_items, k=20, is_lightgcn=True)
    return model, recall, ndcg


if __name__ == "__main__":
    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)

    print("Loading data...")
    n_users, n_items, train, test = load_movielens()
    norm_adj = build_norm_adj(n_users, n_items, train)
    print(f"users={n_users}, items={n_items}")

    # BPR-MF baseline
    _, bpr_recall, bpr_ndcg = train_bpr_mf(train, test, n_users, n_items, epochs=20)

    # LightGCN with layer ablation
    results = {"BPR-MF": (bpr_recall, bpr_ndcg)}
    for n_layers in [1, 2, 3]:
        _, recall, ndcg = train_lightgcn(
            train, test, n_users, n_items, norm_adj, epochs=20, n_layers=n_layers
        )
        results[f"LightGCN-{n_layers}L"] = (recall, ndcg)

    print("\n=== FINAL RESULTS ===")
    print(f"{'Model':<20} {'Recall@20':<12} {'NDCG@20':<12}")
    for name, (r, n) in results.items():
        print(f"{name:<20} {r:<12.4f} {n:<12.4f}")