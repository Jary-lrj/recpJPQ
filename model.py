import numpy as np
import torch
from strategy.svd import SVDAssignmentStrategy


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


# current: svd version
class ItemCode(torch.nn.Module):
    def __init__(self, pq_m, embedding_size, num_items, sequence_length, device):
        super(ItemCode, self).__init__()
        self.device = device
        self.pq_m = pq_m  # 8
        self.sub_embedding_size = embedding_size // self.pq_m  # 48 / 8
        self.item_code_bytes = embedding_size // self.sub_embedding_size  # 8
        self.vals_per_dim = 256
        self.base_type = torch.uint8
        self.item_codes = torch.zeros(
            size=(num_items + 1, self.item_code_bytes), dtype=self.base_type, device=self.device
        )  # trainable?
        self.centroids = torch.nn.Parameter(
            torch.randn(self.item_code_bytes, 256, self.sub_embedding_size, device=self.device)  # (8, 256, 6)
        )
        self.item_codes_strategy = SVDAssignmentStrategy(self.item_code_bytes, num_items, self.device)
        self.sequence_length = sequence_length
        self.num_items = num_items

    def assign_codes(self, train_users):
        code = self.item_codes_strategy.assign(train_users)
        self.item_codes = code

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape
        input_codes = self.item_codes[input_ids].detach().int()  # (256, 200, 8)
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
        return result


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.pq_m = 8
        self.item_code = ItemCode(self.pq_m, args.hidden_units, item_num, args.maxlen, args.device)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(
        self, log_seqs
    ):  # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
        seqs = self.item_code(torch.LongTensor(log_seqs).to(self.dev))  # (256, 200) -> (256, 200, 48)
        # seqs *= self.item_emb.embedding_dim**0.5 # scaling?
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

    # def log2feats(
    #     self, log_seqs
    # ):  # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
    #     seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
    #     seqs *= self.item_emb.embedding_dim**0.5
    #     poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
    #     # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
    #     poss *= log_seqs != 0
    #     seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
    #     seqs = self.emb_dropout(seqs)

    #     tl = seqs.shape[1]  # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     for i in range(len(self.attention_layers)):
    #         seqs = torch.transpose(seqs, 0, 1)
    #         Q = self.attention_layernorms[i](seqs)
    #         mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
    #         # need_weights=False) this arg do not work?
    #         seqs = Q + mha_outputs
    #         seqs = torch.transpose(seqs, 0, 1)

    #         seqs = self.forward_layernorms[i](seqs)
    #         seqs = self.forward_layers[i](seqs)

    #     log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

    #     return log_feats

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

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

    def get_seq_embedding(self, input_ids):
        pass
