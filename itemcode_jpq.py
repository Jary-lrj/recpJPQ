import torch
from strategy.svd import SVDAssignmentStrategy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
import numpy as np


class ItemCodeJPQ(torch.nn.Module):
    def __init__(self, pq_m, embedding_size, num_items, sequence_length, device):
        super(ItemCodeJPQ, self).__init__()
        self.device = device
        self.pq_m = pq_m  # 8
        self.embedding_size = embedding_size
        self.sub_embedding_size = embedding_size // self.pq_m  # 48 / 8
        self.item_code_bytes = embedding_size // self.sub_embedding_size  # 8
        self.vals_per_dim = 128
        self.base_type = torch.int
        self.item_codes = torch.zeros(
            size=(
                num_items, self.item_code_bytes), dtype=self.base_type, device=self.device
        )  # trainable?
        self.centroids = torch.nn.Parameter(
            torch.randn(self.item_code_bytes, self.vals_per_dim,
                        self.sub_embedding_size, device=self.device)
        )
        with torch.no_grad():
            self.centroids[:, 0, :] = 0  # 设置 centroids 的第 0 个索引为全零向量
        self.n_centroids = [self.vals_per_dim] * self.pq_m
        self.item_codes_strategy = SVDAssignmentStrategy(
            self.item_code_bytes, num_items, self.device)
        self.sequence_length = sequence_length
        self.num_items = num_items

    def assign_codes_recJPQ(self, train_users):
        code = self.item_codes_strategy.assign(train_users)
        self.item_codes = code

    def assign_codes_reduced(self, item_embeddings):
        num_items, reduced_dim = item_embeddings.shape

        for i in range(reduced_dim):
            subspace_data = item_embeddings[:, i].cpu(
            ).numpy().reshape(-1, 1)  # Shape: (num_items, 1)
            n_bins = 256  # Ensure bins don't exceed data points
            kbin_discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile")
            cluster_labels = (
                kbin_discretizer.fit_transform(
                    subspace_data).astype(int).flatten()
            )  # Shape: (num_items,)
            self.item_codes[:, i] = torch.from_numpy(
                cluster_labels).to(self.device)

    def get_all_item_embeddings(self):
        codes = self.item_codes[1:]
        embeddings = []
        for i in range(self.item_code_bytes):
            code_indices = codes[:, i].long()
            embedding = self.centroids[i][code_indices]
            embeddings.append(embedding)
        all_embeddings = torch.cat(embeddings, dim=-1)
        all_embeddings = torch.cat(
            [torch.zeros(1, self.embedding_size, device=self.device), all_embeddings], dim=0
        )
        return all_embeddings

    # KMeans-based method
    def assign_codes_KMeans(self, item_embeddings):
        reshaped_embeddings = item_embeddings.reshape(
            self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
        )
        for i in range(self.item_code_bytes):
            subspace_data = reshaped_embeddings[:, i, :].cpu().numpy()
            kmeans = KMeans(
                n_init=10, n_clusters=self.n_centroids[i], random_state=42)
            kmeans.fit(subspace_data)
            cluster_labels = kmeans.predict(subspace_data)
            self.item_codes[:, i] = torch.from_numpy(
                cluster_labels.astype(np.int32)).to(self.device)
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
            if len(centers) < self.vals_per_dim:
                padding = torch.randn(
                    (self.vals_per_dim - len(centers), self.sub_embedding_size))
                centers = torch.cat([centers, padding])
            self.centroids.data[i] = centers.to(self.device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape
        n_centroids = self.n_centroids
        input_codes = self.item_codes[input_ids].detach().int()
        for i in range(self.item_code_bytes):
            input_codes[:, :, i] = torch.clamp(
                input_codes[:, :, i], max=n_centroids[i] - 1)
        code_byte_indices = torch.arange(
            self.item_code_bytes, device=self.device).unsqueeze(0).unsqueeze(0)
        code_byte_indices = code_byte_indices.repeat(
            batch_size, sequence_length, 1)
        n_sub_embeddings = batch_size * sequence_length * self.item_code_bytes
        code_byte_indices_reshaped = code_byte_indices.reshape(
            n_sub_embeddings)
        input_codes_reshaped = input_codes.reshape(n_sub_embeddings)
        indices = torch.stack(
            [code_byte_indices_reshaped, input_codes_reshaped], dim=-1)
        input_sub_embeddings_reshaped = self.centroids[indices[:,
                                                               0], indices[:, 1]]
        result = input_sub_embeddings_reshaped.reshape(
            batch_size, sequence_length, self.item_code_bytes * self.sub_embedding_size
        )
        # Handle number 0 item
        mask = (input_ids == 0).unsqueeze(-1).repeat(1, 1,
                                                     self.item_code_bytes * self.sub_embedding_size)
        result[mask] = 0.0
        return result
