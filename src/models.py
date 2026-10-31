import torch
import torch.nn as nn


class BPRMF(nn.Module):
    """BPR-MF baseline: simple matrix factorization with BPR loss."""
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(dim=1)

    def get_all_scores(self, users, n_items):
        u = self.user_emb(users)
        i = self.item_emb.weight
        return u @ i.T


class LightGCN(nn.Module):
    """LightGCN: simplified GCN for recommendation (He et al., SIGIR 2020)."""
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, norm_adj=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(n_users + n_items, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)

    def forward(self):
        # Initial embeddings for all nodes (users + items)
        ego_embeddings = self.embedding.weight
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            # LightGCN propagation: E^(k+1) = A_norm @ E^(k)
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Layer combination: average of all layers' embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        users_emb = final_embeddings[:self.n_users]
        items_emb = final_embeddings[self.n_users:]
        return users_emb, items_emb

    def get_all_scores(self):
        users_emb, items_emb = self.forward()
        return users_emb @ items_emb.T


def bpr_loss(pos_scores, neg_scores):
    """BPR loss: -log sigmoid(pos - neg)."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()