import torch
from strategy.svd import SVDAssignmentStrategy
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import TruncatedSVD


class ItemCodeDPQ(torch.nn.Module):
    def __init__(self, pq_m, embedding_size, num_items, sequence_length, device):
        super(ItemCodeDPQ, self).__init__()
        self.device = device
        self.pq_m = pq_m  # 8
        self.embedding_size = embedding_size
        self.sub_embedding_size = embedding_size // self.pq_m  # 48 / 8
        self.item_code_bytes = embedding_size // self.sub_embedding_size  # 8
        self.vals_per_dim = 256
        self.base_type = torch.int
        self.item_codes = torch.zeros(
            size=(
                num_items, self.item_code_bytes), dtype=self.base_type, device=self.device
        )  # trainable?
        self.centroids = torch.nn.Parameter(
            torch.randn(self.item_code_bytes, self.vals_per_dim,
                        self.sub_embedding_size, device=self.device)
        )
        self.n_centroids = [self.vals_per_dim] * self.pq_m
        self.item_codes_strategy = SVDAssignmentStrategy(
            self.item_code_bytes, num_items, self.device)
        self.sequence_length = sequence_length
        self.num_items = num_items

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

    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for user, item_set in train_users.items():
            for item in item_set:
                rows.append(user - 1)
                cols.append(item)
                vals.append(1)

        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor([rows, cols]).to(self.device)
        values = torch.FloatTensor(vals).to(self.device)
        shape = (len(train_users), self.num_items)
        matr = torch.sparse_coo_tensor(indices, values, shape).to(self.device)

        print("fitting svd for initial centroids assignments")
        svd = TruncatedSVD(n_components=self.embedding_size)
        svd.fit(matr.cpu().to_dense().numpy())
        item_embeddings = torch.from_numpy(svd.components_).to(self.device)
        item_embeddings = item_embeddings.T
        item_embeddings[0, :] = 0.0
        return item_embeddings

    # KMeans-based method
    def assign_codes_KMeans(self, item_embeddings=None):
        reshaped_embeddings = item_embeddings.reshape(
            self.num_items, self.item_code_bytes, self.sub_embedding_size
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
            self.centroids.data[i] = centers.to(self.device)

        np.save('random.npy', self.item_codes.cpu().numpy())

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

    def reconstruct_loss(self, original_embedding, reconstructed_embedding):
        return torch.nn.functional.mse_loss(reconstructed_embedding, original_embedding)
