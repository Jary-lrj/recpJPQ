import torch
from strategy.svd import SVDAssignmentStrategy
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds


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

    def assign(self, train_users):

        user_item_pairs = [(user - 1, item)
                           for user, items in train_users.items() for item in items]
        user_item_pairs = np.array(user_item_pairs, dtype=np.int64)
        rows, cols = user_item_pairs[:, 0], user_item_pairs[:, 1]
        vals = np.ones(len(rows), dtype=np.float32)

        from scipy.sparse import coo_matrix
        sparse_matr = coo_matrix((vals, (rows, cols)), shape=(
            len(train_users), self.num_items))

        from scipy.sparse.linalg import svds
        U, S, Vt = svds(sparse_matr, k=self.embedding_size)

        item_embeddings = torch.tensor(Vt.T, device=self.device)

        reshaped_embeddings = item_embeddings.view(
            self.num_items, self.item_code_bytes, self.sub_embedding_size)

        min_vals = reshaped_embeddings.min(dim=2, keepdim=True).values
        max_vals = reshaped_embeddings.max(dim=2, keepdim=True).values
        normalized_embeddings = (
            reshaped_embeddings - min_vals) / (max_vals - min_vals + 1e-10)

        noise = torch.normal(
            mean=0.0, std=1e-5, size=normalized_embeddings.shape, device=self.device)
        final_embeddings = (normalized_embeddings +
                            noise).view(self.num_items, -1)

        return final_embeddings

    def assign_codes_soft(self, item_embeddings):
        reshaped_embeddings = item_embeddings.reshape(
            self.num_items, self.item_code_bytes, self.sub_embedding_size
        )

        for i in range(self.item_code_bytes):
            subspace_embeddings = reshaped_embeddings[:, i]
            subspace_centroids = self.centroids[i]

            distances = torch.cdist(
                subspace_embeddings, subspace_centroids, p=2)
            distances = torch.clamp(distances, min=1e-6, max=1e6)

            assignment_probs = F.softmax(-distances, dim=-1)

            hard_assignments = torch.argmax(assignment_probs, dim=-1)

            with torch.no_grad():
                self.item_codes[:, i] = hard_assignments

    def forward(self, input_ids):
        """
        Forward pass for embedding generation.
        input_ids: Tensor of shape (batch_size, sequence_length)
        """
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape

        embeddings = []
        for i in range(self.item_code_bytes):

            item_codes = self.item_codes[input_ids, i].long()

            item_codes = torch.clamp(
                item_codes, min=0, max=self.vals_per_dim - 1)

            item_codes = item_codes.view(-1)

            probs = F.one_hot(
                item_codes, num_classes=self.vals_per_dim).float()

            probs = probs.view(batch_size, sequence_length, self.vals_per_dim)

            embedding = torch.einsum("bsk,kd->bsd", probs, self.centroids[i])
            embeddings.append(embedding)

        result = torch.cat(embeddings, dim=-1)
        mask = (input_ids == 0).unsqueeze(-1).repeat(1, 1, self.embedding_size)
        result[mask] = 0.0
        return result
