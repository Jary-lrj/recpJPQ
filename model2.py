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

        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.item_emb.weight.data[0, :] = 0

        # self.cage = Cage(dim=args.hidden_units, entries=[
        #     256, 256, 256, 256], alpha=1, beta=0.5)

        # self.item_emb = QREmbeddingBag(
        #     num_categories=self.item_num,
        #     embedding_dim=args.hidden_units,
        #     num_collisions=4,
        #     operation="add",
        #     sparse=True,
        #     device=self.dev
        # )

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

        item_embs = self.item_emb(torch.LongTensor(
            item_indices).unsqueeze(0).to(self.dev))  # (U, I, C)

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
        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.5)

        # self.item_emb = QREmbeddingBag(
        #     num_categories=self.item_num,
        #     embedding_dim=args.hidden_units,
        #     num_collisions=4,
        #     operation="add",
        #     sparse=True,
        #     device=self.dev
        # )

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

    def calculate_loss(self, item_seq, item_seq_len):
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
        seq_output = self.calculate_loss(item_seq, item_seq_len)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        return pos_score, neg_score

    def predict(self, user_ids, log_seqs, item_idx):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = log_seqs.shape[1]
        test_item = torch.LongTensor(item_idx).unsqueeze(0).to(self.dev)
        seq_output = self.calculate_loss(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_item).to(self.dev)
        scores = torch.matmul(seq_output, test_item_emb.squeeze(0).T)  # [B]
        return scores

    def full_sort_predict(self, user, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores


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
        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.5)

        # self.item_emb = QREmbeddingBag(
        #     num_categories=self.item_num,
        #     embedding_dim=args.hidden_units,
        #     num_collisions=4,
        #     operation="add",
        #     sparse=True,
        #     device=self.dev
        # )

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
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = log_seqs.shape[1]
        test_item = torch.LongTensor(item_idx).unsqueeze(0).to(self.dev)
        seq_output = self.get_embedding(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_item).to(self.dev)
        scores = torch.matmul(seq_output, test_item_emb.squeeze(0).T)  # [B]
        return scores

    def full_sort_predict(self, users, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(
            self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = (
            torch.matmul(A[:, :, : A.size(1)],
                         self.linear_edge_in(hidden)) + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.size(1): 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(torch.nn.Module):

    def __init__(self, user_num, item_num, args):
        super(SRGNN, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        # load parameters info
        self.embedding_size = args.hidden_units
        self.step = 1

        # define layers and loss
        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.5)
        # self.item_emb = QREmbeddingBag(
        #     num_categories=self.item_num,
        #     embedding_dim=args.hidden_units,
        #     num_collisions=4,
        #     operation="add",
        #     sparse=True,
        #     device=self.dev
        # )
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.embedding_size * 2, self.embedding_size, bias=True
        )

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.dev)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.dev)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.dev)

        return alias_inputs, A, items, mask

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = torch.tensor(gather_index).view(1, 1, 1).expand(
            output.shape[0], -1, output.shape[-1]).to(self.dev)
        output_tensor = output.gather(dim=1, index=gather_index).to(self.dev)
        return output_tensor.squeeze(1)

    def get_embedding(self, item_seq, item_seq_len):
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = self.item_emb(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden *
                      mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
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
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = log_seqs.shape[1]
        test_item = torch.LongTensor(item_idx).unsqueeze(0).to(self.dev)
        seq_output = self.get_embedding(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_item).to(self.dev)
        scores = torch.matmul(seq_output, test_item_emb.squeeze(0).T)  # [B]
        return scores

    def full_sort_predict(self, users, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores


class STAMP(torch.nn.Module):

    def __init__(self, user_num, item_num, args):
        super(STAMP, self).__init__()

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device

        # load parameters info
        self.embedding_size = args.hidden_units

        # define layers and loss
        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.5)
        # self.item_emb = QREmbeddingBag(
        #     num_categories=self.item_num,
        #     embedding_dim=args.hidden_units,
        #     num_collisions=4,
        #     operation="add",
        #     sparse=True,
        #     device=self.dev
        # )
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
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = log_seqs.shape[1]
        test_item = torch.LongTensor(item_idx).unsqueeze(0).to(self.dev)
        seq_output = self.get_embedding(item_seq, item_seq_len)
        test_item_emb = self.item_emb(test_item).to(self.dev)
        scores = torch.matmul(seq_output, test_item_emb.squeeze(0).T)  # [B]
        return scores

    def full_sort_predict(self, users, log_seqs):
        item_seq = log_seqs.to(self.dev)
        item_seq_len = log_seqs.shape[1]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_emb.weight.to(self.dev)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
