import numpy as np
import torch


def mean_shift(self, item_embeddings):
    """使用Mean Shift进行向量量化的编码分配

    Args:
        item_embeddings: 商品的embedding向量
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from scipy.spatial.distance import cdist

    reshaped_embeddings = item_embeddings.reshape(
        self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    )

    # 每个字节最多能表示的类别数
    max_clusters = 256

    for i in range(self.item_code_bytes):
        subspace_data = reshaped_embeddings[:, i, :]

        # 估计带宽(bandwidth)参数
        # quantile越小，产生的簇越多
        bandwidth = estimate_bandwidth(
            subspace_data,
            quantile=0.2,  # 可调参数，控制簇的数量
            n_samples=min(1000, len(subspace_data)),  # 用于估计的样本数
            random_state=42,
        )

        # 如果估计的带宽太小(可能导致过多的簇)，设置一个最小值
        min_bandwidth = np.std(subspace_data) / 10
        bandwidth = max(bandwidth, min_bandwidth)

        # 使用Mean Shift进行聚类
        ms = MeanShift(
            bandwidth=bandwidth,  # 带宽参数
            bin_seeding=True,  # 使用binning技术加速
            cluster_all=True,  # 确保所有点都被分配
            n_jobs=-1,  # 使用所有CPU核心
        )

        cluster_labels = ms.fit_predict(subspace_data)
        cluster_centers = ms.cluster_centers_
        n_clusters = len(cluster_centers)

        # 处理聚类数量
        if n_clusters > max_clusters:
            # 如果簇太多，需要合并一些簇
            print(f"Warning: MeanShift produced {n_clusters} clusters, merging to {max_clusters}")

            # 使用距离矩阵来合并最近的簇
            while n_clusters > max_clusters:
                # 计算簇中心之间的距离
                distances = cdist(cluster_centers, cluster_centers)
                # 将对角线设置为无穷大，避免自己和自己比较
                np.fill_diagonal(distances, np.inf)

                # 找到最近的两个簇
                min_i, min_j = np.unravel_index(distances.argmin(), distances.shape)

                # 合并这两个簇
                # 更新簇标签
                cluster_labels[cluster_labels == min_j] = min_i

                # 重新计算被合并簇的中心
                merged_points = subspace_data[np.logical_or(cluster_labels == min_i, cluster_labels == min_j)]
                new_center = np.mean(merged_points, axis=0)

                # 更新簇中心列表
                cluster_centers = np.delete(cluster_centers, min_j, axis=0)
                cluster_centers[min_i] = new_center

                # 重新映射大于min_j的标签
                cluster_labels[cluster_labels > min_j] -= 1

                n_clusters -= 1

            # 更新簇中心
            centers = cluster_centers

        elif n_clusters < max_clusters:
            # 如果簇太少，保持现状并后续填充
            centers = cluster_centers
        else:
            centers = cluster_centers

        # 确保标签从0开始连续
        unique_labels = np.unique(cluster_labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        cluster_labels = np.array([label_map[label] for label in cluster_labels])

        # 将标签转换为张量并存储
        self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

        # 将簇中心转换为张量并填充到256
        centers = torch.from_numpy(centers).float()
        if len(centers) < 256:
            padding = torch.zeros((256 - len(centers), self.sub_embedding_size))
            centers = torch.cat([centers, padding])

        self.centroids.data[i] = centers.to(self.device)

    def estimate_optimal_bandwidth(self, data, target_clusters=256):
        """估计最优带宽参数的辅助函数"""
        # 二分搜索合适的带宽
        bandwidth_low = np.std(data) / 100
        bandwidth_high = np.std(data) * 2

        for _ in range(10):  # 最多尝试10次
            bandwidth = (bandwidth_low + bandwidth_high) / 2
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(data)
            n_clusters = len(ms.cluster_centers_)

            if n_clusters > target_clusters:
                bandwidth_low = bandwidth
            elif n_clusters < target_clusters:
                bandwidth_high = bandwidth
            else:
                break

        return bandwidth
