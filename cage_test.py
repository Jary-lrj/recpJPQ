import torch
from cage import Cage

tensor = torch.tensor([0.0]*11)
cage = Cage(dim=11, entries=[5, 5])
print(cage(tensor))
