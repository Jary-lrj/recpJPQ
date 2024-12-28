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
import torch.nn.functional as F
from torch.nn import Parameter

from quotient_remainder import QREmbedding as QREmbeddingBag
from cage.cage import Cage


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(
            self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # as Conv1D requires (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        self.embedding_size = args.hidden_units
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        # self.item_emb = torch.nn.Embedding(
        #     self.item_num, args.hidden_units, padding_idx=0)
        # self.item_emb.weight.data[0, :] = 0

        # self.cage = Cage(dim=args.hidden_units, entries=[
        #     256, 128, 64, 32], alpha=1, beta=0.5)

        self.item_emb = QREmbeddingBag(
            num_categories=self.item_num,
            embedding_dim=args.hidden_units,
            num_collisions=4,
            operation="add",
            sparse=True,
            device=self.dev
        )

        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.pos_emb.weight.data[0, :] = 0
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-12)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(
                args.hidden_units, eps=1e-12)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(
                args.hidden_units, eps=1e-12)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(
        self, log_seqs
    ):  # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(
            self.dev))  # (256, 200) -> (256, 200, 48)
        # seqs = self.cage(seqs)
        seqs *= (self.embedding_size) ** 0.5
        poss = np.tile(
            np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= log_seqs != 0
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones(
            (tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask)
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        # only use last QKV classifier, a waste
        final_feat = log_feats[:, -1, :]
        test_items = torch.LongTensor(item_indices).to(
            self.dev)  # Shape: [batch_size, num_items]
        test_item_emb = self.item_emb(test_items)
        scores = torch.bmm(test_item_emb, final_feat.unsqueeze(-1)).squeeze(-1)
        return scores

    def get_seq_embedding(self, input_ids):
        pass

    def get_all_item_embeddings(self):
        ids = torch.arange(0, self.item_num, device=self.dev).long()
        embs = self.item_emb(ids)
        return embs

    def get_cage_item_embeddings(self):
        embs = self.get_all_item_embeddings()
        return self.cage(embs)


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
        # self.item_emb = torch.nn.Embedding(
        #     self.item_num, args.hidden_units, padding_idx=0)
        # self.item_emb.weight.data[0, :] = 0

        # self.cage = Cage(dim=args.hidden_units, entries=[
        #     256, 128, 64, 32], alpha=1, beta=0.25)

        self.item_emb = QREmbeddingBag(
            num_categories=self.item_num,
            embedding_dim=args.hidden_units,
            num_collisions=4,
            operation="add",
            sparse=True,
            device=self.dev
        )

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = torch.tensor(
            gather_index).view(-1, 1, 1).expand(-1, -1, output.shape[-1]).to(self.dev)
        output_tensor = output.gather(dim=1, index=gather_index).to(self.dev)
        return output_tensor.squeeze(1)

    def get_embedding(self, item_seq, item_seq_len):
        item_seq = item_seq.to(self.dev)
        item_seq_emb = self.item_emb(item_seq).to(self.dev)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb).to(self.dev)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output).to(self.dev)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def forward(self, user, log_seqs, pos_seqs, neg_seqs):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = item_seq.shape[1]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        return pos_score, neg_score

    def predict(self, user_ids, log_seqs, item_idx):
        item_seq = torch.LongTensor(log_seqs).to(
            self.dev)  # Shape: [batch_size, seq_len]
        item_seq_len = item_seq.size(1)  # Sequence length
        test_items = torch.LongTensor(item_idx).to(
            self.dev)  # Shape: [batch_size, num_items]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_items)
        scores = torch.bmm(test_item_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores

    def full_sort_predict(self, user, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def get_all_item_embeddings(self):
        ids = torch.arange(0, self.item_num, device=self.dev).long()
        embs = self.item_emb(ids)
        return embs

    def get_cage_item_embeddings(self):
        embs = self.get_all_item_embeddings()
        return self.cage(embs)


class NARM(torch.nn.Module):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self, user_num, item_num, args):
        super(NARM, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        # load parameters info
        self.embedding_size = args.hidden_units
        self.hidden_size = args.hidden_units
        self.n_layers = args.num_blocks
        self.dropout_probs = args.dropout_rate

        # define layers and loss
        # self.item_emb = torch.nn.Embedding(
        #     self.item_num, args.hidden_units, padding_idx=0)
        # self.item_emb.weight.data[0, :] = 0

        # self.cage = Cage(dim=args.hidden_units, entries=[
        #     256, 256, 256, 256], alpha=1, beta=0.5)

        self.item_emb = QREmbeddingBag(
            num_categories=self.item_num,
            embedding_dim=args.hidden_units,
            num_collisions=4,
            operation="add",
            sparse=True,
            device=self.dev
        )

        self.emb_dropout = nn.Dropout(self.dropout_probs)
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs)
        self.b = nn.Linear(2 * self.hidden_size,
                           self.embedding_size, bias=False)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = torch.tensor(gather_index).view(1, 1, 1).expand(
            output.shape[0], -1, output.shape[-1]).to(self.dev)
        output_tensor = output.gather(dim=1, index=gather_index).to(self.dev)
        return output_tensor.squeeze(1)

    def get_embedding(self, item_seq, item_seq_len):
        item_seq = item_seq.to(self.dev)
        item_seq_emb = self.item_emb(item_seq).to(self.dev)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb).to(self.dev)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out).to(self.dev)
        q1 = self.a_1(gru_out).to(self.dev)
        q2 = self.a_2(ht).to(self.dev)
        q2_expand = q2.unsqueeze(1).expand_as(q1).to(self.dev)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand)).to(self.dev)
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1).to(self.dev)
        c_t = torch.cat([c_local, c_global], 1).to(self.dev)
        c_t = self.ct_dropout(c_t).to(self.dev)
        seq_output = self.b(c_t).to(self.dev)
        return seq_output

    def forward(self, user, log_seqs, pos_seqs, neg_seqs):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = item_seq.shape[1]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        seq_output = seq_output.unsqueeze(1)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        return pos_score, neg_score

    def predict(self, user_ids, log_seqs, item_idx):
        item_seq = torch.LongTensor(log_seqs).to(
            self.dev)  # Shape: [batch_size, seq_len]
        item_seq_len = item_seq.size(1)  # Sequence length
        test_items = torch.LongTensor(item_idx).to(
            self.dev)  # Shape: [batch_size, num_items]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_items)
        scores = torch.bmm(test_item_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores

    def full_sort_predict(self, users, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def get_all_item_embeddings(self):
        ids = torch.arange(0, self.item_num, device=self.dev).long()
        embs = self.item_emb(ids)
        return embs

    def get_cage_item_embeddings(self):
        embs = self.get_all_item_embeddings()
        return self.cage(embs)


class Caser(nn.Module):

    def __init__(self, user_num, item_num, args):
        super(Caser, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device
        self.args = args

        # load parameters info
        self.embedding_size = args.hidden_units
        self.n_h = 0
        self.n_v = 1
        self.dropout_prob = args.dropout_rate
        self.max_seq_length = args.maxlen

        # our method
        self.pq_m = args.segment
        # self.item_emb = torch.nn.Embedding(
        #     self.item_num, args.hidden_units, padding_idx=0)
        # self.item_emb.weight.data[0, :] = 0
        self.item_emb = QREmbeddingBag(
            num_categories=self.item_num,
            embedding_dim=args.hidden_units,
            num_collisions=4,
            operation="add",
            sparse=True,
            device=self.dev
        )

        # vertical conv layer
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1)
        )

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.embedding_size),
                )
                for i in lengths
            ]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)  # 200,200
        self.fc2 = nn.Linear(
            self.embedding_size, self.embedding_size
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

    def get_embedding(self, item_seq):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)
        item_seq_emb = self.item_emb(item_seq).unsqueeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        # out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out_v)
        # fully-connected layer
        seq_output = self.ac_fc(self.fc2(self.ac_fc(self.fc1(out))))
        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        return seq_output

    def forward(self, user, log_seqs, pos_seqs, neg_seqs):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = item_seq.shape[1]
        seq_output = self.get_embedding(item_seq)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        seq_output = seq_output.unsqueeze(1)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        return pos_score, neg_score

    def predict(self, user_ids, log_seqs, item_idx):
        item_seq = torch.LongTensor(log_seqs).to(
            self.dev)  # Shape: [batch_size, seq_len]
        test_items = torch.LongTensor(item_idx).to(
            self.dev)  # Shape: [batch_size, num_items]
        seq_output = self.get_embedding(item_seq)
        test_item_emb = self.item_emb(test_items)
        scores = torch.bmm(test_item_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores

    def get_all_item_embeddings(self):
        ids = torch.arange(0, self.item_num, device=self.dev).long()
        embs = self.item_emb(ids)
        return embs


class STAMP(torch.nn.Module):

    def __init__(self, user_num, item_num, args):
        super(STAMP, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        # load parameters info
        self.embedding_size = args.hidden_units

        # define layers and loss
        # self.item_emb = torch.nn.Embedding(
        #     self.item_num, args.hidden_units, padding_idx=0)
        # self.item_emb.weight.data[0, :] = 0

        # self.cage = Cage(dim=args.hidden_units, entries=[
        #     256, 256, 256, 256], alpha=1, beta=0.5)

        self.item_emb = QREmbeddingBag(
            num_categories=self.item_num,
            embedding_dim=args.hidden_units,
            num_collisions=4,
            operation="add",
            sparse=True,
            device=self.dev
        )

        self.w1 = nn.Linear(self.embedding_size,
                            self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size,
                            self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size,
                            self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(
            self.embedding_size), requires_grad=True)
        self.mlp_a = nn.Linear(self.embedding_size,
                               self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size,
                               self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = torch.tensor(gather_index).view(1, 1, 1).expand(
            output.shape[0], -1, output.shape[-1]).to(self.dev)
        output_tensor = output.gather(dim=1, index=gather_index).to(self.dev)
        return output_tensor.squeeze(1)

    def get_embedding(self, item_seq, item_seq_len):
        item_seq_emb = self.item_emb(item_seq)
        last_inputs = self.gather_indexes(item_seq_emb, item_seq_len - 1)
        org_memory = item_seq_emb
        item_seq_len = torch.tensor(item_seq_len).to(self.dev)
        if item_seq_len.dim() == 0:  # 如果是标量，增加一个批量维度
            item_seq_len = item_seq_len.unsqueeze(0)
        ms = torch.div(torch.sum(org_memory, dim=1),
                       item_seq_len.unsqueeze(1).float())
        alpha = self.count_alpha(org_memory, last_inputs, ms)
        vec = torch.matmul(alpha.unsqueeze(1), org_memory)
        ma = vec.squeeze(1) + ms
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        seq_output = hs * ht
        return seq_output

    def count_alpha(self, context, aspect, output):
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(
            -1, timesteps, self.embedding_size
        )
        output_3dim = output.repeat(1, timesteps).view(
            -1, timesteps, self.embedding_size
        )
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha

    def forward(self, user, log_seqs, pos_seqs, neg_seqs):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = item_seq.shape[1]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        seq_output = seq_output.unsqueeze(1)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        return pos_score, neg_score

    def predict(self, user_ids, log_seqs, item_idx):
        item_seq = torch.LongTensor(log_seqs).to(
            self.dev)  # Shape: [batch_size, seq_len]
        item_seq_len = item_seq.size(1)  # Sequence length
        test_items = torch.LongTensor(item_idx).to(
            self.dev)  # Shape: [batch_size, num_items]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_items)
        scores = torch.bmm(test_item_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores

    def full_sort_predict(self, users, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def get_all_item_embeddings(self):
        ids = torch.arange(0, self.item_num, device=self.dev).long()
        embs = self.item_emb(ids)
        return embs

    def get_cage_item_embeddings(self):
        embs = self.get_all_item_embeddings()
        return self.cage(embs)
