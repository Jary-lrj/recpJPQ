import numpy as np
import torch
import torch.nn as nn
from strategy.svd import SVDAssignmentStrategy
from sklearn.cluster import (
    MiniBatchKMeans,
    KMeans,
    DBSCAN,
    MeanShift,
    estimate_bandwidth,
    SpectralClustering,
    AgglomerativeClustering,
    OPTICS,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from utils import check_unique_item
import faiss
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from item_pretrain import train_contrastive_model


class AdaptivePoolingReduction(torch.nn.Module):
    def __init__(self, input_dim, target_dim, pooling_type="mean"):
        super(AdaptivePoolingReduction, self).__init__()
        if pooling_type.lower() == "mean":
            self.pooling = torch.nn.AdaptiveAvgPool1d(target_dim)
        elif pooling_type.lower() == "max":
            self.pooling = torch.nn.AdaptiveMaxPool1d(target_dim)
        else:
            raise ValueError("Pooling type must be 'mean' or 'max'.")

    def forward(self, x):
        x = x.unsqueeze(1)
        pooled_x = self.pooling(x)
        return pooled_x.squeeze(1)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class ItemCode(torch.nn.Module):
    def __init__(self, pq_m, embedding_size, num_items, sequence_length, device):
        super(ItemCode, self).__init__()
        self.device = device
        self.pq_m = pq_m  # 8
        self.embedding_size = embedding_size
        self.sub_embedding_size = embedding_size // self.pq_m  # 48 / 8
        self.item_code_bytes = embedding_size // self.sub_embedding_size  # 8
        self.vals_per_dim = 256
        self.base_type = torch.int
        self.item_codes = torch.zeros(
            size=(num_items, self.item_code_bytes), dtype=self.base_type, device=self.device
        )  # trainable?
        self.centroids = torch.nn.Parameter(
            torch.randn(self.item_code_bytes, self.vals_per_dim, self.sub_embedding_size, device=self.device)
        )
        with torch.no_grad():
            self.centroids[:, 0, :] = 0  # 设置 centroids 的第 0 个索引为全零向量
        self.n_centroids = [self.vals_per_dim] * self.pq_m
        self.item_codes_strategy = SVDAssignmentStrategy(self.item_code_bytes, num_items, self.device)
        self.sequence_length = sequence_length
        self.num_items = num_items

    def assign_codes_recJPQ(self, train_users):
        code = self.item_codes_strategy.assign(train_users)
        self.item_codes = code
        print(self.item_codes[0])

    def assign_codes_reduced(self, item_embeddings):
        num_items, reduced_dim = item_embeddings.shape

        for i in range(reduced_dim):
            subspace_data = item_embeddings[:, i].cpu().numpy().reshape(-1, 1)  # Shape: (num_items, 1)
            n_bins = 256  # Ensure bins don't exceed data points
            kbin_discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            cluster_labels = (
                kbin_discretizer.fit_transform(subspace_data).astype(int).flatten()
            )  # Shape: (num_items,)
            self.item_codes[:, i] = torch.from_numpy(cluster_labels).to(self.device)

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
            kmeans = KMeans(n_init=10, n_clusters=self.n_centroids[i], random_state=42)
            kmeans.fit(subspace_data)
            cluster_labels = kmeans.predict(subspace_data)
            self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.int32)).to(self.device)
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
            if len(centers) < self.vals_per_dim:
                padding = torch.randn((self.vals_per_dim - len(centers), self.sub_embedding_size))
                centers = torch.cat([centers, padding])
            self.centroids.data[i] = centers.to(self.device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape
        n_centroids = self.n_centroids
        input_codes = self.item_codes[input_ids].detach().int()
        for i in range(self.item_code_bytes):
            input_codes[:, :, i] = torch.clamp(input_codes[:, :, i], max=n_centroids[i] - 1)
        code_byte_indices = torch.arange(self.item_code_bytes, device=self.device).unsqueeze(0).unsqueeze(0)
        code_byte_indices = code_byte_indices.repeat(batch_size, sequence_length, 1)
        n_sub_embeddings = batch_size * sequence_length * self.item_code_bytes
        code_byte_indices_reshaped = code_byte_indices.reshape(n_sub_embeddings)
        input_codes_reshaped = input_codes.reshape(n_sub_embeddings)
        indices = torch.stack([code_byte_indices_reshaped, input_codes_reshaped], dim=-1)
        input_sub_embeddings_reshaped = self.centroids[indices[:, 0], indices[:, 1]]
        result = input_sub_embeddings_reshaped.reshape(
            batch_size, sequence_length, self.item_code_bytes * self.sub_embedding_size
        )
        # Handle number 0 item
        mask = (input_ids == 0).unsqueeze(-1).repeat(1, 1, self.item_code_bytes * self.sub_embedding_size)
        result[mask] = 0.0
        return result


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-12)

        # self.reduce = AdaptivePoolingReduction(args.hidden_units, args.segment, "mean")
        self.pq_m = args.segment
        # self.item_embeddings_padding_type = args.type
        # self.dataset_name = "beauty"
        # self.item_embeddings = torch.load(
        #     f"./glove_embedding/{self.dataset_name}/{self.pq_m}_seg/{self.item_embeddings_padding_type}.pt"
        # )
        self.item_code = ItemCode(self.pq_m, args.hidden_units, self.item_num, args.maxlen, args.device)
        # self.reduced_item_embeddings = self.reduce(self.item_embeddings)
        # self.pretrain_item_embeddings = train_contrastive_model(
        #     self.item_embeddings, f"./data/{self.dataset_name}.txt", args.hidden_units
        # )
        # self.item_code.assign_codes(self.reduced_item_embeddings)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-12)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-12)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(
        self, log_seqs
    ):  # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
        seqs = self.item_code(torch.LongTensor(log_seqs).to(self.dev))  # (256, 200) -> (256, 200, 48)
        seqs *= (self.pq_m * 50) ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= log_seqs != 0
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_code(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_code(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_code(torch.LongTensor(item_indices).unsqueeze(0).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

    def get_seq_embedding(self, input_ids):
        pass


class GRU4Rec(torch.nn.Module):

    def __init__(self, user_num, item_num, args):
        super(GRU4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        # load parameters info
        self.embedding_size = args.hidden_units
        self.hidden_size = args.hidden_units
        self.num_layers = args.num_blocks
        self.dropout_prob = args.dropout_rate

        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # our method
        self.reduce = AdaptivePoolingReduction(args.hidden_units, args.segment, "mean")
        self.pq_m = args.segment
        self.item_embeddings_padding_type = args.type
        self.dataset_name = "beauty"
        self.item_embeddings = torch.load(
            f"./glove_embedding/{self.dataset_name}/{self.pq_m}_seg/{self.item_embeddings_padding_type}.pt"
        )
        self.item_code = ItemCode(self.pq_m, args.hidden_units, self.item_num, args.maxlen, args.device)
        self.reduced_item_embeddings = self.reduce(self.item_embeddings)
        # self.pretrain_item_embeddings = train_contrastive_model(
        #     self.item_embeddings, f"./data/{self.dataset_name}.txt", args.hidden_units
        # )
        self.item_code.assign_codes_reduced(self.reduced_item_embeddings)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = torch.tensor(gather_index).view(-1, 1, 1).expand(-1, -1, output.shape[-1]).to(self.dev)
        output_tensor = output.gather(dim=1, index=gather_index).to(self.dev)
        return output_tensor.squeeze(1)

    def calculate_loss(self, item_seq, item_seq_len):
        item_seq = item_seq.to(self.dev)
        item_seq_emb = self.item_code(item_seq).to(self.dev)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb).to(self.dev)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output).to(self.dev)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def forward(self, user, log_seqs, pos_seqs, neg_seqs):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = item_seq.shape[1]
        seq_output = self.calculate_loss(item_seq, item_seq_len)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_code(pos_items)
        neg_items_emb = self.item_code(neg_items)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        return pos_score, neg_score

    def predict(self, user_ids, log_seqs, item_idx):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = log_seqs.shape[1]
        test_item = torch.LongTensor(item_idx).unsqueeze(0)
        seq_output = self.calculate_loss(item_seq, item_seq_len)
        test_item_emb = self.item_code(test_item).to(self.dev)
        scores = torch.matmul(seq_output, test_item_emb.squeeze(0).T)  # [B]
        return scores

    def full_sort_predict(self, user, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_code.weight.to(self.dev)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
