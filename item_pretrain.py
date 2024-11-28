import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm
import random


class ItemDataset(Dataset):
    def __init__(self, item_embeddings, pairs, device):
        self.item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32).to(device)
        self.pairs = pairs
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, positive = self.pairs[idx]
        return self.item_embeddings[anchor], self.item_embeddings[positive]


class ContrastiveModel(nn.Module):
    def __init__(self, embedding_size, hidden_size=128, dropout_rate=0.2):
        super(ContrastiveModel, self).__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x + residual)  # Residual connection
        return F.normalize(x, dim=1)


def contrastive_loss(anchor, positive, temperature=0.5):
    batch_size = anchor.size(0)
    labels = torch.arange(batch_size).to(anchor.device)
    similarities = F.cosine_similarity(anchor.unsqueeze(1), positive.unsqueeze(0), dim=2)
    similarities /= temperature
    loss = F.cross_entropy(similarities, labels)
    return loss


def generate_pairs(user_item_matrix, start_idx=0, length=15):
    pairs = []
    for user, items in tqdm(user_item_matrix.items(), desc="Generating pairs"):

        if len(items) < 2:
            continue

        if start_idx + length > len(items):
            continue

        for i in range(start_idx, start_idx + length - 1):
            pairs.append((items[i], items[i + 1]))
    return pairs


# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_contrastive_model(
    glove_embeddings, user_item_path, embedding_size=300, batch_size=256, lr=1e-3, epochs=200, device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    glove_embeddings = glove_embeddings.to(device)
    user_item = np.loadtxt(user_item_path)

    item_embeddings = normalize(glove_embeddings.cpu().numpy())
    user_item_matrix = {}
    for row in user_item:
        user = int(row[0])
        item = int(row[1])
        if user not in user_item_matrix:
            user_item_matrix[user] = []
        user_item_matrix[user].append(item)

    min_length = min(len(items) for items in user_item_matrix.values())
    print(f"Minimum length in user_item_matrix: {min_length}")

    pairs = generate_pairs(user_item_matrix)
    dataset = ItemDataset(item_embeddings, pairs, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ContrastiveModel(embedding_size=embedding_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for anchor, positive in progress_bar:
            anchor, positive = anchor.to(device), positive.to(device)
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            loss = contrastive_loss(anchor_out, positive_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    with torch.no_grad():
        optimized_embeddings = model(torch.tensor(item_embeddings, dtype=torch.float32).to(device))

    return optimized_embeddings


# Example usage:
# optimized_embeddings = train_contrastive_model("./glove_embedding/ml-1m/6_seg/normal_random.pt", "./data/ml-1m.txt")
