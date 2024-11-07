import numpy as np
import torch
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from utils import check_unique_item
import faiss
from concurrent.futures import ThreadPoolExecutor


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
        self.base_type = torch.int
        self.item_codes = torch.zeros(
            size=(num_items + 1, self.item_code_bytes), dtype=self.base_type, device=self.device
        )  # trainable?
        self.centroids = torch.nn.Parameter(
            torch.randn(self.item_code_bytes, 256, self.sub_embedding_size, device=self.device)  # (8, 256, 6)
        )
        self.n_centroids = [256] * self.pq_m
        self.item_codes_strategy = SVDAssignmentStrategy(self.item_code_bytes, num_items, self.device)
        self.sequence_length = sequence_length
        self.num_items = num_items

    def assign_codes(self, train_users):
        code = self.item_codes_strategy.assign(train_users)
        print(f"Duplicate Rows: {check_unique_item(code)}")
        self.item_codes = code

    # KMeans-based method
    # def assign_codes(self, item_embeddings):
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )
    #     n_centroids = self.n_centroids
    #     for i in range(self.item_code_bytes):
    #         subspace_data = reshaped_embeddings[:, i, :]
    #         kmeans = KMeans(n_init=10, n_clusters=n_centroids[i], random_state=42)
    #         cluster_labels = kmeans.fit_predict(subspace_data)
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.int32)).to(self.device)
    #         centers = torch.from_numpy(kmeans.cluster_centers_).float()

    #         self.centroids.data[i] = centers.to(self.device)
    #     print(f"Duplicate Rows: {check_unique_item(self.centroids)}")

    # # Hierarchical clustering-based method
    # def assign_codes(self, item_embeddings):
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )
    #     n_centroids = self.n_centroids

    #     for i in range(self.item_code_bytes):
    #         subspace_data = reshaped_embeddings[:, i, :].cpu().numpy()

    #         # 1. 计算距离矩阵
    #         distances = pdist(subspace_data)

    #         # 2. 执行层次聚类
    #         # method可以是: 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    #         linkage_matrix = linkage(distances, method="ward")

    #         # 3. 获取聚类标签
    #         cluster_labels = (
    #             fcluster(linkage_matrix, t=n_centroids[i], criterion="maxclust") - 1  # 指定聚类数量
    #         )  # -1使标签从0开始

    #         # 4. 计算聚类中心
    #         centers = []
    #         for j in range(n_centroids[i]):
    #             cluster_points = subspace_data[cluster_labels == j]
    #             if len(cluster_points) > 0:
    #                 center = cluster_points.mean(axis=0)
    #             else:
    #                 # 如果某个类别为空，用随机点初始化
    #                 center = np.random.randn(self.sub_embedding_size)
    #             centers.append(center)
    #         centers = np.array(centers)

    #         # 5. 保存编码和中心点
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.int32)).to(self.device)
    #         centers = torch.from_numpy(centers).float()

    #         # 6. 填充到256
    #         if n_centroids[i] < 256:
    #             padding = torch.zeros((256 - n_centroids[i], self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         self.centroids.data[i] = centers.to(self.device)

    # DBSCAN-based method
    # def assign_codes(self, item_embeddings):
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )

    #     max_clusters = 256  # 每个字节最多能表示的类别数

    #     for i in range(self.item_code_bytes):
    #         subspace_data = reshaped_embeddings[:, i, :]
    #         dbscan = DBSCAN(eps=0.5, min_samples=10)  # 邻域半径+最小样本数
    #         cluster_labels = dbscan.fit_predict(subspace_data)

    #         # 处理噪声点(标签为-1的点)
    #         noise_mask = cluster_labels == -1
    #         if noise_mask.any():
    #             # 将噪声点分配给最近的非噪声簇
    #             noise_points = subspace_data[noise_mask]
    #             non_noise_labels = np.unique(cluster_labels[~noise_mask])

    #             if len(non_noise_labels) > 0:
    #                 # 计算每个噪声点到各个簇中心的距离
    #                 cluster_centers = np.array(
    #                     [subspace_data[cluster_labels == label].mean(axis=0) for label in non_noise_labels]
    #                 )

    #                 # 为噪声点分配最近的簇
    #                 distances = cdist(noise_points, cluster_centers)
    #                 nearest_cluster = non_noise_labels[distances.argmin(axis=1)]
    #                 cluster_labels[noise_mask] = nearest_cluster
    #             else:
    #                 # 如果所有点都是噪声点，将它们归为一类
    #                 cluster_labels[noise_mask] = 0

    #         # 重新映射标签使其连续且从0开始
    #         unique_labels = np.unique(cluster_labels)
    #         label_map = {old: new for new, old in enumerate(unique_labels)}
    #         cluster_labels = np.array([label_map[label] for label in cluster_labels])

    #         # 确保聚类数不超过256
    #         if len(np.unique(cluster_labels)) > max_clusters:
    #             # 如果超过256个簇，需要合并一些簇
    #             # 这里使用K-means将DBSCAN的结果重新聚类为256类
    #             kmeans = KMeans(n_clusters=max_clusters, random_state=42)
    #             cluster_centers = np.array(
    #                 [
    #                     subspace_data[cluster_labels == label].mean(axis=0)
    #                     for label in np.unique(cluster_labels)
    #                 ]
    #             )
    #             new_labels = kmeans.fit_predict(cluster_centers)
    #             # 映射回原始数据点
    #             cluster_labels = new_labels[cluster_labels]
    #             centers = kmeans.cluster_centers_
    #         else:
    #             # 计算每个簇的中心
    #             centers = np.array(
    #                 [
    #                     subspace_data[cluster_labels == label].mean(axis=0)
    #                     for label in range(len(np.unique(cluster_labels)))
    #                 ]
    #             )

    #         # 将标签转换为张量并存储
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

    #         # 将簇中心转换为张量并填充到256
    #         centers = torch.from_numpy(centers).float()
    #         if len(centers) < 256:
    #             padding = torch.zeros((256 - len(centers), self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         self.centroids.data[i] = centers.to(self.device)

    # Mean Shift-based method
    # def assign_codes(self, item_embeddings):
    #     """使用Mean Shift进行向量量化的编码分配

    #     Args:
    #         item_embeddings: 商品的embedding向量
    #     """
    #     # 将PyTorch张量转换为NumPy数组进行处理
    #     reshaped_embeddings = (
    #         item_embeddings.reshape(self.num_items + 1, self.item_code_bytes, self.sub_embedding_size)
    #         .cpu()
    #         .numpy()
    #     )

    #     # 每个字节最多能表示的类别数
    #     max_clusters = 256

    #     for i in range(self.item_code_bytes):
    #         subspace_data = reshaped_embeddings[:, i, :]

    #         # 计算标准差时使用numpy
    #         std_dev = np.std(subspace_data)

    #         # 估计带宽(bandwidth)参数
    #         bandwidth = estimate_bandwidth(
    #             subspace_data,
    #             quantile=0.2,  # 可调参数，控制簇的数量
    #             n_samples=min(1000, len(subspace_data)),  # 用于估计的样本数
    #             random_state=42,
    #         )

    #         # 如果估计的带宽太小，设置一个最小值
    #         min_bandwidth = std_dev / 10
    #         bandwidth = max(bandwidth, min_bandwidth)

    #         # 使用Mean Shift进行聚类
    #         ms = MeanShift(
    #             bandwidth=bandwidth,  # 带宽参数
    #             bin_seeding=True,  # 使用binning技术加速
    #             cluster_all=True,  # 确保所有点都被分配
    #             n_jobs=-1,  # 使用所有CPU核心
    #         )

    #         cluster_labels = ms.fit_predict(subspace_data)
    #         cluster_centers = ms.cluster_centers_
    #         n_clusters = len(cluster_centers)

    #         # 处理聚类数量
    #         if n_clusters > max_clusters:
    #             print(f"Warning: MeanShift produced {n_clusters} clusters, merging to {max_clusters}")

    #             # 使用距离矩阵来合并最近的簇
    #             while n_clusters > max_clusters:
    #                 # 计算簇中心之间的距离
    #                 distances = cdist(cluster_centers, cluster_centers)
    #                 # 将对角线设置为无穷大，避免自己和自己比较
    #                 np.fill_diagonal(distances, np.inf)

    #                 # 找到最近的两个簇
    #                 min_i, min_j = np.unravel_index(distances.argmin(), distances.shape)

    #                 # 合并这两个簇
    #                 cluster_labels[cluster_labels == min_j] = min_i

    #                 # 重新计算被合并簇的中心
    #                 merged_points = subspace_data[
    #                     np.logical_or(cluster_labels == min_i, cluster_labels == min_j)
    #                 ]
    #                 new_center = np.mean(merged_points, axis=0)

    #                 # 更新簇中心列表
    #                 cluster_centers = np.delete(cluster_centers, min_j, axis=0)
    #                 cluster_centers[min_i] = new_center

    #                 # 重新映射大于min_j的标签
    #                 cluster_labels[cluster_labels > min_j] -= 1

    #                 n_clusters -= 1

    #             # 更新簇中心
    #             centers = cluster_centers

    #         elif n_clusters < max_clusters:
    #             # 如果簇太少，保持现状并后续填充
    #             centers = cluster_centers
    #         else:
    #             centers = cluster_centers

    #         # 确保标签从0开始连续
    #         unique_labels = np.unique(cluster_labels)
    #         label_map = {old: new for new, old in enumerate(unique_labels)}
    #         cluster_labels = np.array([label_map[label] for label in cluster_labels])

    #         # 将标签转换为张量并存储
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

    #         # 将簇中心转换为张量并填充到256
    #         centers = torch.from_numpy(centers).float()
    #         if len(centers) < 256:
    #             padding = torch.zeros((256 - len(centers), self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         self.centroids.data[i] = centers.to(self.device)

    # def estimate_optimal_bandwidth(self, data, target_clusters=256):
    #     """估计最优带宽参数的辅助函数"""
    #     # 确保使用numpy数组
    #     if torch.is_tensor(data):
    #         data = data.cpu().numpy()

    #     # 二分搜索合适的带宽
    #     bandwidth_low = np.std(data) / 100
    #     bandwidth_high = np.std(data) * 2

    #     for _ in range(10):  # 最多尝试10次
    #         bandwidth = (bandwidth_low + bandwidth_high) / 2
    #         ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #         ms.fit(data)
    #         n_clusters = len(ms.cluster_centers_)

    #         if n_clusters > target_clusters:
    #             bandwidth_low = bandwidth
    #         elif n_clusters < target_clusters:
    #             bandwidth_high = bandwidth
    #         else:
    #             break

    #     return bandwidth

    # Spectral Clustering-based method
    # def assign_codes(self, item_embeddings):
    #     # 重塑嵌入向量维度
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )

    #     # 每层的聚类数量
    #     n_clusters = [256, 256, 256, 256]

    #     for i in range(self.item_code_bytes):
    #         # 提取当前子空间的数据
    #         subspace_data = reshaped_embeddings[:, i, :]

    #         # 初始化并训练Spectral Clustering
    #         # 使用nearest_neighbors方法构建亲和矩阵以提高效率
    #         clustering = SpectralClustering(
    #             n_clusters=n_clusters[i],
    #             n_init=3,
    #             random_state=42,
    #             affinity="nearest_neighbors",  # 使用KNN构建亲和矩阵
    #             n_neighbors=100,  # KNN的邻居数量
    #             assign_labels="kmeans",  # 最后一步用kmeans聚类特征向量
    #         )

    #         # 获取聚类标签
    #         cluster_labels = clustering.fit_predict(subspace_data)

    #         # 将标签转换为uint8类型并存储
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

    #         # 计算每个簇的中心点
    #         centers = []
    #         for label in range(n_clusters[i]):
    #             mask = cluster_labels == label
    #             if np.any(mask):
    #                 center = subspace_data[mask].mean(axis=0)
    #             else:
    #                 # 如果某个簇为空，用随机点初始化
    #                 center = subspace_data[np.random.choice(len(subspace_data))].copy()
    #             centers.append(center)

    #         centers = torch.from_numpy(np.stack(centers)).float()

    #         # 如果聚类数量少于256，进行填充
    #         if n_clusters[i] < 256:
    #             padding = torch.zeros((256 - n_clusters[i], self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         # 更新中心点
    #         self.centroids.data[i] = centers.to(self.device)

    # GMM-based method
    # def assign_codes(self, item_embeddings):
    #     # 重塑嵌入向量维度
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )

    #     # 每层的高斯混合成分数量
    #     n_components = [256, 256, 256, 256]

    #     for i in range(self.item_code_bytes):
    #         # 提取当前子空间的数据
    #         subspace_data = reshaped_embeddings[:, i, :]

    #         # 初始化并训练GMM
    #         gmm = GaussianMixture(
    #             n_components=n_components[i],
    #             n_init=3,
    #             random_state=42,
    #             covariance_type="diag",  # 使用对角协方差矩阵以提高效率
    #         )

    #         # 获取聚类标签
    #         cluster_labels = gmm.fit_predict(subspace_data)

    #         # 将标签转换为uint8类型并存储
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

    #         # 获取聚类中心(均值向量)
    #         centers = torch.from_numpy(gmm.means_).float()

    #         # 如果组件数量少于256,进行填充
    #         if n_components[i] < 256:
    #             padding = torch.zeros((256 - n_components[i], self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         # 更新中心点
    #         self.centroids.data[i] = centers.to(self.device)

    # Agglomerative Clustering-based method
    # def assign_codes(self, item_embeddings):
    #     # 重塑嵌入向量维度
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )

    #     # 每层的聚类数量
    #     n_clusters = [256, 256, 256, 256]

    #     for i in range(self.item_code_bytes):
    #         # 提取当前子空间的数据
    #         subspace_data = reshaped_embeddings[:, i, :]

    #         # 初始化并训练Agglomerative Clustering
    #         clustering = AgglomerativeClustering(
    #             n_clusters=n_clusters[i],
    #             linkage="ward",  # 使用Ward链接方法，最小化类内方差
    #             metric="euclidean",  # Ward方法只能使用欧氏距离
    #         )

    #         # 获取聚类标签
    #         cluster_labels = clustering.fit_predict(subspace_data)

    #         # 将标签转换为uint8类型并存储
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

    #         # 计算聚类中心
    #         centers = []
    #         for label in range(n_clusters[i]):
    #             mask = cluster_labels == label
    #             if np.any(mask):
    #                 center = subspace_data[mask].mean(axis=0)
    #             else:
    #                 # 如果出现空簇，使用随机点
    #                 center = subspace_data[np.random.choice(len(subspace_data))].copy()
    #             centers.append(center)

    #         centers = torch.from_numpy(np.stack(centers)).float()

    #         # 如果聚类数量少于256，进行填充
    #         if n_clusters[i] < 256:
    #             padding = torch.zeros((256 - n_clusters[i], self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         # 更新中心点
    #         self.centroids.data[i] = centers.to(self.device)

    # OPTICS-based
    # def assign_codes(self, item_embeddings):
    #     # 重塑嵌入向量维度
    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )

    #     # 目标聚类数量
    #     target_clusters = 256

    #     for i in range(self.item_code_bytes):
    #         # 提取当前子空间的数据并转换为numpy数组进行处理
    #         subspace_data = reshaped_embeddings[:, i, :].cpu().numpy()

    #         # 初始化OPTICS
    #         optics = OPTICS(
    #             min_samples=5,  # 定义核心点的最小邻居数
    #             max_eps=np.inf,  # 最大邻域半径
    #             metric="euclidean",
    #             cluster_method="xi",  # 使用xi方法进行聚类提取
    #             n_jobs=-1,  # 使用所有可用CPU
    #         )

    #         # 拟合OPTICS模型并获取聚类标签
    #         cluster_labels = optics.fit_predict(subspace_data)

    #         # 处理噪声点（标签为-1的点）
    #         noise_mask = cluster_labels == -1
    #         valid_labels = cluster_labels[~noise_mask]

    #         # 如果聚类数量不等于256，需要进行调整
    #         unique_clusters = np.unique(valid_labels)
    #         n_clusters = len(unique_clusters)

    #         if n_clusters > target_clusters:
    #             # 如果聚类太多，合并最相似的簇
    #             centers = []
    #             for label in unique_clusters:
    #                 mask = cluster_labels == label
    #                 if np.any(mask):
    #                     center = subspace_data[mask].mean(axis=0)
    #                     centers.append(center)
    #             centers = np.array(centers)

    #             # 计算簇间距离
    #             distances = cdist(centers, centers)
    #             np.fill_diagonal(distances, np.inf)

    #             # 合并最相近的簇直到达到目标数量
    #             new_labels = cluster_labels.copy()
    #             while len(np.unique(new_labels)) > target_clusters:
    #                 idx1, idx2 = np.unravel_index(distances.argmin(), distances.shape)
    #                 new_labels[new_labels == unique_clusters[idx2]] = unique_clusters[idx1]
    #                 distances[idx2, :] = np.inf
    #                 distances[:, idx2] = np.inf

    #             cluster_labels = new_labels

    #         elif n_clusters < target_clusters:
    #             # 如果聚类太少，对大的簇进行细分
    #             new_labels = cluster_labels.copy()
    #             current_max_label = new_labels.max()

    #             for label in range(n_clusters):
    #                 if len(np.unique(new_labels)) >= target_clusters:
    #                     break

    #                 mask = new_labels == label
    #                 cluster_points = subspace_data[mask]

    #                 if len(cluster_points) > 2:
    #                     # 对大簇使用KMeans进行细分
    #                     n_sub_clusters = min(
    #                         len(cluster_points) // 2, target_clusters - len(np.unique(new_labels)) + 1
    #                     )

    #                     if n_sub_clusters > 1:
    #                         sub_clustering = KMeans(n_clusters=n_sub_clusters, n_init=3).fit_predict(
    #                             cluster_points
    #                         )

    #                         new_labels[mask] = sub_clustering + current_max_label + 1
    #                         current_max_label += n_sub_clusters

    #             cluster_labels = new_labels

    #         # 确保标签从0开始连续
    #         unique_labels = np.unique(cluster_labels)
    #         label_map = {old: new for new, old in enumerate(unique_labels)}
    #         cluster_labels = np.array([label_map[label] for label in cluster_labels])

    #         # 将标签转换为uint8并存储
    #         self.item_codes[:, i] = torch.from_numpy(cluster_labels.astype(np.uint8)).to(self.device)

    #         # 计算最终的聚类中心
    #         centers = []
    #         for label in range(target_clusters):
    #             mask = cluster_labels == label
    #             if np.any(mask):
    #                 center = subspace_data[mask].mean(axis=0)
    #             else:
    #                 # 对于空簇，使用随机点
    #                 random_idx = np.random.choice(len(subspace_data))
    #                 center = subspace_data[random_idx].copy()
    #             centers.append(center)

    #         # 将centers转换为PyTorch张量
    #         centers = torch.tensor(np.stack(centers), dtype=torch.float32).to(self.device)

    #         # 如果中心点数量少于256，进行填充
    #         if len(centers) < 256:
    #             padding = torch.zeros((256 - len(centers), self.sub_embedding_size), device=self.device)
    #             centers = torch.cat([centers, padding])

    #         # 更新中心点
    #         self.centroids.data[i] = centers

    # def assign_codes(self, item_embeddings):

    #     reshaped_embeddings = item_embeddings.reshape(
    #         self.num_items + 1, self.item_code_bytes, self.sub_embedding_size
    #     )

    #     # 确保数据为float32类型
    #     if item_embeddings.dtype != torch.float32:
    #         reshaped_embeddings = reshaped_embeddings.float()

    #     def process_subspace(i):
    #         subspace_data = reshaped_embeddings[:, i, :].cpu().numpy()
    #         subspace_data = np.ascontiguousarray(subspace_data)
    #         n_centroids_i = self.n_centroids[i]

    #         try:
    #             # 如果GPU可用，使用GPU版本
    #             res = faiss.StandardGpuResources()
    #             config = faiss.GpuIndexFlatConfig()
    #             config.useFloat16 = False
    #             config.device = 0

    #             # 创建GPU kmeans
    #             kmeans = faiss.Kmeans(
    #                 d=self.sub_embedding_size,  # 维度
    #                 k=n_centroids_i,  # 聚类中心数量
    #                 niter=20,  # 迭代次数
    #                 nredo=3,  # 重复次数
    #                 gpu=True,  # 使用GPU
    #                 gpu_id=0,  # GPU设备ID
    #                 seed=42,  # 随机种子
    #             )

    #         except Exception as e:
    #             print(f"GPU initialization failed, falling back to CPU: {e}")
    #             # 如果GPU不可用，使用CPU版本
    #             kmeans = faiss.Kmeans(
    #                 d=self.sub_embedding_size, k=n_centroids_i, niter=20, nredo=3, gpu=False, seed=42
    #             )

    #         # 训练
    #         kmeans.train(subspace_data)

    #         # 获取聚类结果
    #         _, cluster_labels = kmeans.index.search(subspace_data, 1)
    #         cluster_labels = cluster_labels.reshape(-1)

    #         # 获取中心点
    #         centers = torch.from_numpy(kmeans.centroids).float()

    #         # 填充处理
    #         if n_centroids_i < 256:
    #             padding = torch.zeros((256 - n_centroids_i, self.sub_embedding_size))
    #             centers = torch.cat([centers, padding])

    #         return {"labels": torch.from_numpy(cluster_labels.astype(np.int32)), "centers": centers}

    #     # 并行处理所有子空间
    #     try:
    #         with ThreadPoolExecutor() as executor:
    #             results = list(executor.map(process_subspace, range(self.item_code_bytes)))

    #         # 更新结果
    #         for i, result in enumerate(results):
    #             self.item_codes[:, i] = result["labels"].to(self.device)
    #             self.centroids.data[i] = result["centers"].to(self.device)

    #         print(self.item_codes)

    #     except Exception as e:
    #         # 如果并行处理失败，回退到串行处理
    #         print(f"Parallel processing failed, falling back to serial processing: {e}")
    #         for i in range(self.item_code_bytes):
    #             result = process_subspace(i)
    #             self.item_codes[:, i] = result["labels"].to(self.device)
    #             self.centroids.data[i] = result["centers"].to(self.device)

    # def forward(self, input_ids):
    #     input_ids = input_ids.to(self.device)
    #     batch_size, sequence_length = input_ids.shape
    #     input_codes = self.item_codes[input_ids].detach().int()  # (256, 200, 8)
    #     code_byte_indices = torch.arange(self.item_code_bytes, device=self.device).unsqueeze(0).unsqueeze(0)
    #     code_byte_indices = code_byte_indices.repeat(batch_size, sequence_length, 1)
    #     n_sub_embeddings = batch_size * sequence_length * self.item_code_bytes
    #     code_byte_indices_reshaped = code_byte_indices.reshape(n_sub_embeddings)
    #     input_codes_reshaped = input_codes.reshape(n_sub_embeddings)
    #     indices = torch.stack([code_byte_indices_reshaped, input_codes_reshaped], dim=-1)
    #     input_sub_embeddings_reshaped = self.centroids[indices[:, 0], indices[:, 1]]
    #     result = input_sub_embeddings_reshaped.reshape(
    #         batch_size, sequence_length, self.item_code_bytes * self.sub_embedding_size
    #     )
    #     return result

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape
        n_centroids = self.n_centroids
        input_codes = self.item_codes[input_ids].detach().int()
        for i in range(self.item_code_bytes):
            input_codes[:, :, i] = torch.clamp(input_codes[:, :, i], max=n_centroids[i] - 1)
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

    def get_item_embedding(self, item_id):
        with torch.no_grad():
            item_code = self.item_codes[item_id].detach().int()
            indices = torch.stack([torch.arange(self.item_code_bytes, device=self.device), item_code], dim=-1)
            item_embedding = self.centroids[indices[:, 0], indices[:, 1]]
            return item_embedding.reshape(-1)


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
        # self.pretrain_embeddings_padding_type = "normal_random"
        # self.dataset_name = "beauty"
        # self.pretrain_embeddings = torch.load(
        #     f"./glove_embedding/{self.dataset_name}/{self.pq_m}_seg/{self.pretrain_embeddings_padding_type}.pt"
        # )
        self.item_code = ItemCode(self.pq_m, args.hidden_units, item_num, args.maxlen, args.device)
        # self.item_code.assign_codes(self.pretrain_embeddings)

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
        seqs *= (self.pq_m * 50) ** 0.5
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

    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
    #     log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits  # pos_pred, neg_pred

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_code(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_code(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    # def predict(self, user_ids, log_seqs, item_indices):  # for inference
    #     log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

    #     final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

    #     item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     # preds = self.pos_sigmoid(logits) # rank same item list for different users

    #     return logits  # preds # (U, I)

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_code(torch.LongTensor(item_indices).unsqueeze(0).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

    def get_seq_embedding(self, input_ids):
        pass
