import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import seaborn as sns
from typing import Union, Optional, List
import os
import torch
import time


def visualize_embedding(
    embedding: Union[torch.Tensor, np.ndarray],
    output_filename: str,
    segment_size: int = 50,
    figsize: tuple = (20, 15),
    dpi: int = 300,
    entropy_bins: int = 110,
    cmap_entropy: str = "YlOrRd",
    cmap_corr: str = "coolwarm",
    save_statistics: bool = True,
    device: str = "cpu",
) -> dict:

    # Convert input to numpy array if it's a torch tensor
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().to(device)
        embedding_np = embedding.cpu().numpy()
    else:
        embedding_np = embedding

    # Input validation
    if embedding_np.ndim != 2:
        raise ValueError("Embedding must be a 2D array/tensor")

    n_samples, n_features = embedding_np.shape
    if n_features % segment_size != 0:
        raise ValueError(
            f"Number of features ({n_features}) must be divisible by segment_size ({segment_size})"
        )

    n_segments = n_features // segment_size

    # Split embedding into segments
    segments = np.split(embedding_np, n_segments, axis=1)

    # Create figure
    plt.figure(figsize=figsize)

    # 1. Calculate and plot segment entropies
    def calculate_segment_entropy(segment):
        """
        计算一个段中所有 50 维向量的熵，返回段的平均熵。
        """
        vector_entropies = []
        for vector in segment:  # 遍历每个 50 维向量
            hist, _ = np.histogram(vector, bins=entropy_bins, density=True)
            hist = hist + 1e-10  # 避免 log(0)
            vector_entropy = entropy(hist)
            vector_entropies.append(vector_entropy)

        # 返回该段所有向量熵的平均值
        return np.mean(vector_entropies)

    # 计算每段的平均熵值
    segment_entropies = [calculate_segment_entropy(seg) for seg in segments]

    # 绘制条形图
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(1, n_segments + 1), segment_entropies)
    plt.title("Segment Entropies (Average per Vector)", fontsize=12, pad=10)
    plt.xlabel("Segment Index")
    plt.ylabel("Average Entropy")

    # 在条形顶部添加标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f"{height:.2f}", ha="center", va="bottom")

    plt.show()

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f"{height:.2f}", ha="center", va="bottom")

    # 2. Calculate and plot dimension entropy boxplot
    dimension_entropies = []
    for segment in segments:
        dim_entropies = []
        for dim in range(segment.shape[1]):
            hist, _ = np.histogram(
                segment[:, dim], bins=entropy_bins, density=True)
            hist = hist + 1e-10
            dim_entropies.append(entropy(hist))
        dimension_entropies.append(dim_entropies)

    plt.subplot(2, 2, 2)
    plt.boxplot(dimension_entropies)
    plt.title("Dimension Entropy Distribution per Segment", fontsize=12, pad=10)
    plt.xlabel("Segment Index")
    plt.ylabel("Entropy")
    plt.grid(True, linestyle="--", alpha=0.7)

    # 3. Plot dimension entropy heatmap
    entropy_matrix = np.array(dimension_entropies)

    plt.subplot(2, 2, 3)
    sns.heatmap(
        entropy_matrix.T, cmap=cmap_entropy, xticklabels=range(n_segments), yticklabels=range(segment_size)
    )
    plt.title("Dimension Entropy Heatmap", fontsize=12, pad=10)
    plt.xlabel("Segment Index")
    plt.ylabel("Dimension Index")

    # 4. Plot segment correlation matrix
    segment_means = np.array([np.mean(seg, axis=1) for seg in segments]).T
    correlation_matrix = np.corrcoef(segment_means, rowvar=False)

    plt.subplot(2, 2, 4)
    sns.heatmap(
        correlation_matrix,
        cmap=cmap_corr,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        xticklabels=range(n_segments),
        yticklabels=range(n_segments),
    )
    plt.title("Segment Correlation Matrix", fontsize=12, pad=10)
    plt.xlabel("Segment Index")
    plt.ylabel("Segment Index")

    # Adjust layout and save
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_filename) if os.path.dirname(
        output_filename) else ".", exist_ok=True)

    # Save figure
    plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
    plt.close()

    # Compute statistics
    statistics = {
        "segment_entropies": segment_entropies,
        "dimension_entropies": dimension_entropies,
        "correlation_matrix": correlation_matrix.tolist(),
        "summary": {
            "mean_entropy": np.mean(segment_entropies),
            "std_entropy": np.std(segment_entropies),
            "max_correlation": np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
            "min_correlation": np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
            "mean_correlation": np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
        },
    }

    # Optionally save statistics to file
    if save_statistics:
        stats_filename = os.path.splitext(output_filename)[0] + "_stats.txt"
        with open(stats_filename, "w") as f:
            f.write("Embedding Analysis Statistics\n")
            f.write("===========================\n\n")
            f.write(f"Input type: {type(embedding).__name__}\n")
            f.write(f"Input shape: {embedding_np.shape}\n\n")
            f.write("Segment Entropies:\n")
            for i, entropy_val in enumerate(segment_entropies):
                f.write(f"Segment {i}: {entropy_val:.3f}\n")
            f.write("\nSummary Statistics:\n")
            for key, value in statistics["summary"].items():
                f.write(f"{key}: {value:.3f}\n")

    return statistics


def plot_loss_curve(model, loss_list, dataset, segment, type_):

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label="Loss", color="blue")

    plt.title(
        f"Loss Over Time for {dataset} Dataset with {segment} Segment and {type_} Type")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    directory = "loss_fig"
    if not os.path.isdir(directory):
        os.makedirs(directory)

    fname = f"loss_{model}_{dataset}_{segment}_segment_{type_}_{timestamp}.png"

    save_path = os.path.join(directory, fname)
    plt.savefig(save_path)

    return save_path


if __name__ == "__main__":
    # # Example with PyTorch tensor
    # sample_embedding_torch = torch.load("./item_embeddings.pth")

    # # Visualize and get statistics using PyTorch tensor
    # stats_torch = visualize_embedding(
    #     embedding=sample_embedding_torch,
    #     output_filename="embedding_analysis_torch.png",
    #     segment_size=50,
    #     figsize=(20, 15),
    #     dpi=300,
    #     save_statistics=True,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    sample_loss = [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.13, 0.12, 0.11]

    # Plot with default settings
    save_path = plot_loss_curve(
        loss_list=sample_loss, dataset="beauty", segment="6", type_="SASREC", args=None
    )
    print(f"Plot saved at: {save_path}")
