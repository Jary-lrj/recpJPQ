import torch

linear = torch.nn.Embedding(10, 5)
ids = torch.arange(0, 10)
print(linear(ids))
