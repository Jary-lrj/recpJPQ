# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Quotient-Remainder Trick
#
# Description: Applies quotient remainder-trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Hao-Jun Michael Shi, Dheevatsa Mudigere, Maxim Naumov, Jiyan Yang,
# "Compositional Embeddings Using Complementary Partitions for Memory-Efficient
# Recommendation Systems", CoRR, arXiv:1909.02107, 2019


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class QREmbedding(nn.Module):
    __constants__ = [
        "num_categories",
        "embedding_dim",
        "num_collisions",
        "operation",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
    ]

    def __init__(
        self,
        num_categories,
        embedding_dim,
        num_collisions,
        operation="mult",
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device='cpu'
    ):
        super(QREmbedding, self).__init__()

        assert operation in ["concat", "mult", "add"], "Not valid operation!"

        self.num_categories = num_categories
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions
        self.operation = operation
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.device = device

        if self.operation == "add" or self.operation == "mult":
            assert (
                self.embedding_dim[0] == self.embedding_dim[1]
            ), "Embedding dimensions do not match!"

        self.num_embeddings = [
            int(np.ceil(num_categories / num_collisions)),
            num_collisions,
        ]

        if _weight is None:
            self.weight_q = nn.Embedding(
                self.num_embeddings[0], self.embedding_dim[0], device=self.device)
            self.weight_r = nn.Embedding(
                self.num_embeddings[1], self.embedding_dim[1], device=self.device)
            self.reset_parameters()
        else:
            self.weight_q = nn.Embedding.from_pretrained(_weight[0])
            self.weight_r = nn.Embedding.from_pretrained(_weight[1])

    def reset_parameters(self):
        nn.init.uniform_(self.weight_q.weight, -np.sqrt(1 /
                         self.num_categories), np.sqrt(1 / self.num_categories))
        nn.init.uniform_(self.weight_r.weight, -np.sqrt(1 /
                         self.num_categories), np.sqrt(1 / self.num_categories))

    def forward(self, input):
        input_q = (input / self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()

        embed_q = self.weight_q(input_q).to(self.device)
        embed_r = self.weight_r(input_r).to(self.device)

        if self.operation == "concat":
            embed = torch.cat((embed_q, embed_r), dim=-1)
        elif self.operation == "add":
            embed = embed_q + embed_r
        elif self.operation == "mult":
            embed = embed_q * embed_r

        return embed

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        return s.format(**self.__dict__)


if __name__ == "__main__":
    # Example usage
    qr_embedding = QREmbedding(
        num_categories=1000,
        embedding_dim=200,
        num_collisions=4,
        operation="mult",
        sparse=True
    )

    input = torch.LongTensor([[0, 1, 2, 3]]).to('cuda:0')
    print(input.shape)
    embed = qr_embedding(input)
    print(embed[0])
    print(embed.shape)
