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
import math
import copy


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


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(
            mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(
            mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(
            mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": torch.nn.functional.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                   for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SASRec(nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        # load parameters info
        self.n_layers = args.num_blocks
        self.n_heads = args.num_heads
        self.hidden_size = args.hidden_units  # same as embedding_size
        self.inner_size = args.hidden_units
        self.hidden_dropout_prob = args.dropout_rate
        self.attn_dropout_prob = args.dropout_rate
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12

        self.user_num = user_num
        self.item_num = item_num + 1
        self.dev = args.device
        self.args = args

        # define layers and loss
        self.item_emb = nn.Embedding(
            self.item_num, self.hidden_size, padding_idx=0
        )
        self.item_emb.weight.data[0, :] = 0
        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.25, vocab_size=self.item_num)
        self.position_embedding = nn.Embedding(
            args.maxlen, self.hidden_size)
        self.position_embedding.weight.data[0, :] = 0

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.pq_m = args.segment

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(
            extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def get_embedding(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_emb(item_seq)
        item_emb, _ = self.cage(item_emb)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = torch.tensor(
            gather_index).view(-1, 1, 1).expand(-1, -1, output.shape[-1]).to(self.dev)
        output_tensor = output.gather(dim=1, index=gather_index).to(self.dev)
        return output_tensor.squeeze(1)

    def forward(self, user, log_seqs, pos_seqs, neg_seqs):
        item_seq = torch.LongTensor(log_seqs).to(self.dev)
        item_seq_len = item_seq.shape[1]
        seq_output = self.get_embedding(item_seq, item_seq_len)
        pos_items = torch.LongTensor(pos_seqs).to(self.dev)
        neg_items = torch.LongTensor(neg_seqs).to(self.dev)
        pos_items_emb = self.item_emb(pos_items)
        pos_items_emb, _ = self.cage(pos_items_emb)
        neg_items_emb = self.item_emb(neg_items)
        neg_items_emb, _ = self.cage(neg_items_emb)
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
        test_item_emb, _ = self.cage(test_item_emb)
        scores = torch.bmm(test_item_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores

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
        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.item_emb.weight.data[0, :] = 0

        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.25, vocab_size=self.item_num)

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

    def get_embedding(self, item_seq, item_seq_len):
        item_seq = item_seq.to(self.dev)
        item_seq_emb = self.item_emb(item_seq).to(self.dev)
        item_seq_emb, _ = self.cage(item_seq_emb)
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
        pos_items_emb, loss2 = self.cage(pos_items_emb)
        neg_items_emb = self.item_emb(neg_items)
        neg_items_emb, loss3 = self.cage(neg_items_emb)
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
        test_item_emb, _ = self.cage(test_item_emb)
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
        cage_embs, _ = self.cage(embs)
        return cage_embs


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
        self.item_emb.weight.data[0, :] = 0

        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.25, vocab_size=self.item_num)

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
        item_seq_emb, _ = self.cage(item_seq_emb)
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
        pos_items_emb, _ = self.cage(pos_items_emb)
        neg_items_emb = self.item_emb(neg_items)
        neg_items_emb, _ = self.cage(neg_items_emb)
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
        test_item_emb, _ = self.cage(test_item_emb)
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

        self.item_emb = torch.nn.Embedding(
            self.item_num, args.hidden_units, padding_idx=0)
        self.item_emb.weight.data[0, :] = 0

        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.25, vocab_size=self.item_num)

        # self.item_emb = QREmbeddingBag(
        #     num_categories=self.item_num,
        #     embedding_dim=args.hidden_units,
        #     num_collisions=4,
        #     operation="add",
        #     sparse=True,
        #     device=self.dev
        # )

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
        item_seq_emb, _ = self.cage(item_seq_emb)
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
        pos_items_emb, _ = self.cage(pos_items_emb)
        neg_items_emb = self.item_emb(neg_items)
        neg_items_emb, _ = self.cage(neg_items_emb)
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
        test_item_emb, _ = self.cage(test_item_emb)
        scores = torch.bmm(test_item_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores

    def get_all_item_embeddings(self):
        ids = torch.arange(0, self.item_num, device=self.dev).long()
        embs = self.item_emb(ids)
        return embs

    def get_cage_item_embeddings(self):
        embs = self.get_all_item_embeddings()
        return self.cage(embs)


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
        self.item_emb.weight.data[0, :] = 0

        self.cage = Cage(dim=args.hidden_units, entries=[
            256, 256, 256, 256], alpha=1, beta=0.5, vocab_size=self.item_num)

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
        item_seq_emb, _ = self.cage(item_seq_emb)
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
        pos_items_emb, _ = self.cage(pos_items_emb)
        neg_items_emb = self.item_emb(neg_items)
        neg_items_emb, _ = self.cage(neg_items_emb)
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
        test_item_emb, _ = self.cage(test_item_emb)
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
