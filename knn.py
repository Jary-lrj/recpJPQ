import numpy as np
import faiss
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
# 设置随机种子
np.random.seed(42)


def generate_item_codes(n=10000, d=4):
    """
    随机生成 n 个 item 的 item code，每个 item code 长度为 4。
    """
    return np.random.randn(n, d).astype('float32')

# 2. 使用 FAISS 构建 KNN 索引并检索最近邻


def knn_search(item_codes, k=10):
    """
    使用 FAISS 构建 KNN 索引并返回最近邻索引和距离。
    """
    n, d = item_codes.shape
    index = faiss.IndexFlatL2(d)  # 基于 L2 距离的 FAISS 索引
    index.add(item_codes)  # 添加 item codes 到索引
    distances, indices = index.search(item_codes, k + 1)  # 包括自身，排除自身
    return distances[:, 1:], indices[:, 1:]  # 排除自身的距离和索引

# 3. 分析最近邻距离分布


def analyze_knn_distances(distances):
    """
    分析最近邻距离的分布，包括均值、标准差，并绘制直方图。
    """
    avg_distances = np.mean(distances, axis=1)  # 每个 item 到其 k 近邻的平均距离
    overall_mean = np.mean(avg_distances)  # 总体平均距离
    overall_std = np.std(avg_distances)    # 总体标准差

    print(f"平均邻居距离: {overall_mean:.4f}")
    print(f"邻居距离标准差: {overall_std:.4f}")

    # 绘制直方图
    plt.figure(figsize=(8, 5))
    plt.hist(avg_distances, bins=50, alpha=0.7)
    plt.xlabel("Average Distance to k-Nearest Neighbors")
    plt.ylabel("Frequency")
    plt.title("Distribution of k-NN Distances")
    plt.savefig("knn_distances.png")

    return avg_distances
# 4. 主函数


def t_test_distances(distances1, distances2):
    """
    执行独立样本 t 检验，验证两个样本的距离分布是否存在显著性差异。
    """
    t_stat, p_value = ttest_ind(distances1, distances2, equal_var=False)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value}")
    if p_value < 0.05:
        print("结果显著：两个分布之间存在显著性差异。")
    else:
        print("结果不显著：两个分布之间没有显著性差异。")


def main():
    dataset = "movies"
    item_code_cate = np.load(f'./item_code/{dataset}/cate.npy')
    item_code_random = np.load(f'./item_code/{dataset}/random.npy')
    item_code_cate = item_code_cate / 255.0
    item_code_random = item_code_random / 255.0
    distance_cate = analyze_knn_distances(item_code_cate)
    distance_random = analyze_knn_distances(item_code_random)
    t_test_distances(distance_cate, distance_random)


if __name__ == "__main__":
    main()
