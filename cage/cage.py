import torch
from torch import nn
from torch.nn import functional as F

from cage.common import TransformLayer, DecoderLayer, CageQuantization, CageClassification, CageModule


class Cage(CageModule):
    """
    Cascade Clusterer
    """

    def __init__(
            self,
            dim,
            entries=None,  # ex. [100, 10]
            alpha=1,  # alpha
            beta=0.25,  # beta,
            vocab_size=1
    ):
        super().__init__()

        if entries is not None and not isinstance(entries, list):
            entries = str(entries)
            entries = [int(x) for x in entries.split('-')]

        self.embed_dim = dim
        self.vocab_size = vocab_size
        self.num_layers = -1  # type: int
        self.cluster_sizes = entries  # type: list[int]
        self.weighted_add = alpha
        self.commitment_cost = beta
        self.layer_connect = True
        self.layer_loss = True

        assert entries is not None, "cluster_sizes must be specified"

        self.set_cluster_size()

        # construct codebooks
        # ex. [nn.Embedding(4, D), nn.Embedding(2, D)]
        self.codebooks = nn.ModuleList()
        for i in range(self.num_layers):
            self.codebooks.append(nn.Embedding(self.cluster_sizes[i], dim))

        # layers for classification
        self.transform_layer = TransformLayer(
            embed_dim=self.embed_dim,
            activation_function='relu',
        )
        self.decoder_layer = DecoderLayer(
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size,
        )
        # decoder layers for each layer
        # ex. [DecoderLayer(D, 4), DecoderLayer(D, 2)]
        self.codebook_decoders = nn.ModuleList()
        for i in range(self.num_layers):
            self.codebook_decoders.append(DecoderLayer(
                embed_dim=self.embed_dim,
                vocab_size=self.cluster_sizes[i],
            ))

    def set_cluster_size(self):
        self.num_layers = len(self.cluster_sizes)
        assert self.num_layers >= 0 and isinstance(
            self.num_layers, int), "num_layers must be a non-negative integer"

        top_cluster_size = int(
            self.vocab_size ** (1.0 / (self.num_layers + 1)) + 0.5)
        self.cluster_sizes = [top_cluster_size]  # ex. [2]
        for i in range(self.num_layers - 1):
            self.cluster_sizes.append(top_cluster_size * self.cluster_sizes[i])
        self.cluster_sizes = self.cluster_sizes[::-1]  # ex. [4, 2]

    def quantize(
            self,
            embeds,
            with_loss=True,
    ) -> CageQuantization:
        compare_embeds = embeds  # for loss calculation

        shape = embeds.shape  # 原始形状
        embeds = embeds.view(-1, self.embed_dim)  # 展平为二维张量 [B * ..., D]
        qembeds = []
        qindices = []

        for i in range(self.num_layers):
            is_zero_vector = (embeds == 0).all(dim=-1, keepdim=True)
            dist = torch.cdist(embeds, self.codebooks[i].weight, p=2)
            indices = torch.argmin(dist, dim=-1).unsqueeze(1)
            placeholder = torch.zeros(
                indices.shape[0], self.cluster_sizes[i], device=embeds.device)
            placeholder.scatter_(1, indices, 1)
            inner_embeds = torch.matmul(
                placeholder, self.codebooks[i].weight).view(embeds.shape)
            inner_embeds = torch.where(
                is_zero_vector, torch.zeros_like(inner_embeds), inner_embeds)

            qembeds.append(inner_embeds.view(shape))
            qindices.append(indices.view(shape[:-1]))

        # 更新嵌入层
        if self.layer_connect:
            embeds = inner_embeds

        output = CageQuantization(qembeds, indices=qindices)
        embeds = embeds.view(shape)
        if output.mean.any():
            output.mean += embeds * self.weighted_add

        if not with_loss:
            return output

        q_loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        for i in range(self.num_layers):
            q_loss += F.mse_loss(qembeds[i].detach(), compare_embeds) * self.commitment_cost \
                + F.mse_loss(qembeds[i], compare_embeds.detach())
            if self.layer_connect:
                compare_embeds = qembeds[i]
        output.loss = q_loss

        return output.mean, output.loss

    def classify(
            self,
            embeds,
            indices=None,
    ) -> CageClassification:
        embeds = self.transform_layer(embeds)
        scores = self.decoder_layer(embeds)

        cls_loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        if indices:
            for i in range(self.num_layers):
                layer_scores = self.codebook_decoders[i](embeds)
                if self.layer_loss:
                    cls_loss += F.cross_entropy(layer_scores,
                                                indices[i].view(-1), reduction='mean')

        return CageClassification(scores, layer_loss=cls_loss)

    def __call__(self, *args, **kwargs):
        return self.quantize(*args, **kwargs)


if __name__ == "main":
    tensor = torch.randn(10)
    cage = Cage(dim=10, entries=[10, 10])
    print(cage(tensor))
