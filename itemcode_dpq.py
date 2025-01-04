import torch
from strategy.svd import SVDAssignmentStrategy
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix


class ItemCodeDPQ(nn.Module):
    def __init__(self, pq_m, embedding_size, num_items, sequence_length, device):
        super(ItemCodeDPQ, self).__init__()
        self.device = device
        self.pq_m = pq_m  # 8
        self.embedding_size = embedding_size
        self.sub_embedding_size = embedding_size // self.pq_m  # 48 / 8
        self.item_code_bytes = embedding_size // self.sub_embedding_size  # 8
        self.vals_per_dim = 256
        self.base_type = torch.float  # Changed to float for differentiability

        # Trainable item codes (initialized randomly, now trainable)
        self.item_codes = nn.Parameter(
            torch.randn(num_items, self.item_code_bytes,
                        device=self.device)
        )

        # Trainable centroids
        self.centroids = nn.Parameter(
            torch.randn(self.item_code_bytes, self.vals_per_dim,
                        self.sub_embedding_size, device=self.device)
        )
        self.sequence_length = sequence_length
        self.num_items = num_items
        self.original_embeddings = None

    def get_all_item_embeddings(self):
        """Generate all item embeddings using hard assignments and centroids."""
        embeddings = []

        for i in range(self.item_code_bytes):

            hard_assignments = self.item_codes[:, i].long()

            assignment_probs = F.one_hot(
                hard_assignments, num_classes=self.vals_per_dim).float()

            embedding = torch.matmul(assignment_probs, self.centroids[i])

            embeddings.append(embedding)

        all_embeddings = torch.cat(embeddings, dim=-1)
        return all_embeddings

    def assign(self, train_users, pretrain_emb_path=None):
        train_items = set(item for items in train_users.values()
                          for item in items)
        all_item_embeddings = torch.zeros(
            self.num_items, self.embedding_size, device=self.device
        )

        if pretrain_emb_path is None:
            user_item_pairs = [
                (user - 1, item) for user, items in train_users.items() for item in items
            ]
            user_item_pairs = np.array(user_item_pairs, dtype=np.int64)
            rows, cols = user_item_pairs[:, 0], user_item_pairs[:, 1]
            vals = np.ones(len(rows), dtype=np.float32)
            sparse_matr = coo_matrix((vals, (rows, cols)), shape=(
                len(train_users), self.num_items))
            U, S, Vt = svds(sparse_matr, k=self.embedding_size)

            item_embeddings = torch.tensor(Vt.T, device=self.device)
            all_item_embeddings = item_embeddings
            all_item_embeddings[0, :] = 0.0
        else:
            item_embeddings = torch.load(pretrain_emb_path).to(self.device)
            for item in train_items:
                all_item_embeddings[item] = item_embeddings[item]

        return all_item_embeddings

    def assign_codes_soft(self, item_embeddings):

        reshaped_embeddings = item_embeddings.reshape(
            self.num_items, self.item_code_bytes, self.sub_embedding_size)

        with torch.no_grad():
            for i in range(self.item_code_bytes):
                subspace_embeddings = reshaped_embeddings[:, i].to(
                    torch.float32)
                subspace_centroids = self.centroids[i].to(torch.float32)
                distances = torch.cdist(
                    subspace_embeddings, subspace_centroids, p=2)
                probs = F.softmax(-distances, dim=-1)
                self.item_codes[:, i] = torch.multinomial(
                    probs, num_samples=1).squeeze()

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape

        embeddings = []
        for i in range(self.item_code_bytes):
            item_codes = self.item_codes[input_ids, i].long()
            probs = F.one_hot(
                item_codes, num_classes=self.vals_per_dim).float()
            probs = probs.view(batch_size, sequence_length, self.vals_per_dim)
            embedding = torch.einsum("bsk,kd->bsd", probs, self.centroids[i])
            embeddings.append(embedding)

        result = torch.cat(embeddings, dim=-1)
        mask = (input_ids == 0).unsqueeze(-1).repeat(1, 1, self.embedding_size)
        result[mask] = 0.0

        return result
