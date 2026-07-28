import numpy as np
import torch
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional, Dict, Union
from collections import defaultdict


# =============================================
# 工具函数：Tensor 转 NumPy
# =============================================
def convert_tensor_to_numpy(x):
    """将PyTorch张量转换为NumPy数组（处理梯度和设备问题）"""
    if torch.is_tensor(x):
        if x.requires_grad:
            x = x.detach()
        if x.is_cuda:
            return x.cpu().numpy()
        return x.numpy()
    return x


# =============================================
# 多视角降维（PCA、NMF、ICA）
# =============================================
def multi_view_dimension_reduction(features, num_components=20):
    """对输入特征进行多视角降维（PCA、NMF、ICA）
    
    如果所有降维方法都失败，将使用原始特征作为降级策略
    """
    features = convert_tensor_to_numpy(features)
    n_samples, n_feats = features.shape
    # 确保降维维度合法（不超过样本数、特征数，且至少为1）
    comps = min(num_components, n_feats, max(1, n_samples - 1))

    views = []
    view_names = []

    # 1. PCA
    try:
        pca = PCA(n_components=comps, random_state=42)
        views.append(pca.fit_transform(features))
        view_names.append("PCA")
    except Exception as e:
        print(f"[Warning] PCA failed: {e}. Skipping PCA.")

    # 2. NMF
    try:
        X_nonneg = features - np.min(features) + 1e-8  # 平移到非负区间，加1e-8避免0
        nmf = NMF(n_components=comps, random_state=42, max_iter=1000)
        views.append(nmf.fit_transform(X_nonneg))
        view_names.append("NMF")
    except Exception as e:
        print(f"[Warning] NMF failed: {e}. Skipping NMF.")

    # 3. ICA
    try:
        ica = FastICA(n_components=comps, random_state=42, max_iter=1000)
        views.append(ica.fit_transform(features))
        view_names.append("ICA")
    except Exception as e:
        print(f"[Warning] ICA failed: {e}. Skipping ICA.")

    # 降级策略：如果所有降维方法都失败，使用原始特征
    if not views:
        print("[Warning] All dimension reduction methods failed. Using original features as fallback.")
        # 使用原始特征，如果维度太高则进行简单的PCA降维
        if n_feats > num_components:
            try:
                # 尝试最简单的PCA配置
                pca = PCA(n_components=min(comps, n_samples - 1), random_state=42, svd_solver='randomized')
                views.append(pca.fit_transform(features))
                view_names.append("PCA_Fallback")
            except Exception as e:
                print(f"[Error] Even fallback PCA failed: {e}. Using truncated original features.")
                # 最后的降级：截取前num_components维
                views.append(features[:, :min(comps, n_feats)])
                view_names.append("Truncated_Features")
        else:
            views.append(features)
            view_names.append("Original_Features")

    return views, view_names


# =============================================
# 主函数：高置信样本 + 类别兜底 + 轮转平衡采样
# =============================================
def select_high_confidence_samples_by_voting(
        features,
        class_probs,
        num_components=20,
        num_clusters=10,
        seed=None,
        balance_sample_total=None,
        balance_replace=False,
        fallback_topk=20  # 兜底填补每个缺失类别的样本数
):  # 明确返回类型：(最终样本索引, 最终伪标签)
    """
    通过多视角聚类投票选择高置信度样本，并进行类别平衡采样

    参数:
        features: 输入特征 (张量或数组)
        class_probs: 类别概率分布 (张量或数组，形状为 [n_samples, n_classes])
        num_components: 降维后的维度
        num_clusters: 聚类数量
        seed: 随机种子，保证可复现性
        balance_sample_total: 总采样数量，None则自动平衡
        balance_replace: 样本不足时是否允许有放回采样
        fallback_topk: 为缺失类别补充的样本数量

    返回:
        最终采样的样本索引列表和对应的伪标签列表
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 转换为NumPy数组
    features_np = convert_tensor_to_numpy(features)
    class_probs_np = convert_tensor_to_numpy(class_probs)
    n_samples, n_classes = class_probs_np.shape

    # 基本参数校验
    if n_classes <= 1:
        raise ValueError(f"类别数必须大于1，当前为{n_classes}")
    if num_clusters <= 0:
        raise ValueError(f"聚类数量必须为正数，当前为{num_clusters}")
    if fallback_topk <= 0:
        raise ValueError(f"兜底样本数必须为正数，当前为{fallback_topk}")

    # --------------------------
    # Step 1. 多视角降维
    # --------------------------
    views, view_names = multi_view_dimension_reduction(features_np, num_components=num_components)
    print(f"使用的降维视角：{view_names}")
    n_views = len(views)

    # --------------------------
    # Step 2. 单视角聚类 + 类簇→类别映射
    # --------------------------
    sample_votes = []
    for view_data in views:
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(view_data)
        view_votes = np.full(n_samples, -1, dtype=int)

        for c in range(num_clusters):
            cluster_mask = (cluster_labels == c)
            if not np.any(cluster_mask):  # 跳过空簇
                continue

            # 取簇内所有样本的预测类别（基于类别概率）
            cluster_probs = class_probs_np[cluster_mask]
            cluster_preds = np.argmax(cluster_probs, axis=1)

            # 多数投票确定簇的代表类别
            majority_class = np.bincount(cluster_preds).argmax()
            view_votes[cluster_mask] = majority_class

        sample_votes.append(view_votes)

    # --------------------------
    # Step 3. 投票 + 置信度
    # --------------------------
    high_confidence_indices = []
    pseudo_labels = np.full(n_samples, -1, dtype=int)
    confidence_scores = {}

    for sample_idx in range(n_samples):
        votes = [sample_votes[view_idx][sample_idx] for view_idx in range(len(view_names))]
        valid_votes = [v for v in votes if v != -1]
        V = len(valid_votes)
        unique_votes, vote_counts = np.unique(valid_votes, return_counts=True)
        C = len(unique_votes)
        max_vote_count = np.max(vote_counts) if len(vote_counts) > 0 else 0

        if V == 0:
            continue

        if C == 1:
            assigned_class = unique_votes[0]
            confidence_score = round(1.0 * (V / 3.0), 2)
        else:
            assigned_class = unique_votes[np.argmax(vote_counts)]
            confidence_score = round(0.3 * (max_vote_count / V), 2)
            pass

        # 记录高置信样本
        pseudo_labels[sample_idx] = assigned_class
        confidence_scores[sample_idx] = confidence_score
        high_confidence_indices.append(sample_idx)

    # 去重高置信样本索引（防止重复添加）
    high_confidence_indices = list(set(high_confidence_indices))

    # --------------------------
    # Step 4. 平衡采样 缺失类别兜底
    # --------------------------
    # 1. 按类别分组高置信样本（含兜底样本）
    class_samples = defaultdict(list)
    for idx in high_confidence_indices:
        cls = pseudo_labels[idx]
        if cls != -1:  # 排除无效类别
            class_samples[cls].append((idx, confidence_scores[idx]))

    # 检查是否所有类别都有样本
    for cls in range(n_classes):
        if cls not in class_samples or len(class_samples[cls]) == 0:
            print(f"紧急处理：类别 {cls + 1} 无样本，强制补充该类别概率最高的样本")
            # 提取该类别的所有样本概率，按概率降序排序
            cls_probs = class_probs_np[:, cls]
            # 过滤：仅保留概率≥0.1的样本
            valid_mask = cls_probs >= 0.1
            valid_indices = np.where(valid_mask)[0]
            valid_probs = cls_probs[valid_mask]

            # 从有效样本中按概率降序排序
            sorted_valid_indices = valid_indices[np.argsort(valid_probs)[::-1]]
            top_k = min(fallback_topk, len(sorted_valid_indices))
            top_indices = sorted_valid_indices[:top_k]
            for idx in top_indices:
                class_samples[cls].append((idx, float(round(cls_probs[idx], 2))))

    # 2. 确定总采样数量
    if balance_sample_total is None:
        balance_sample_total = n_classes * 10  # 默认每个类别10个样本

    # 3. 计算所有样本的采样概率
    all_sample_indices = []
    all_sample_confidences = []
    all_sample_labels = []

    # 收集所有样本信息
    for cls in range(n_classes):
        samples = class_samples[cls]
        for idx, conf in samples:
            all_sample_indices.append(idx)
            all_sample_confidences.append(conf)
            all_sample_labels.append(cls)

    # 统计每个类别在最终标签中的样本数量
    final_class_counts = defaultdict(int)
    for cls in range(n_classes):
        final_class_counts[cls] = len(class_samples[cls])

    # 计算每个样本的初始采样概率（基于置信度）
    conf_array = np.array(all_sample_confidences)
    initial_probs = conf_array / np.sum(conf_array) if np.sum(conf_array) > 0 else np.ones(len(conf_array)) / len(
        conf_array)

    # 根据最终标签分布调整采样概率
    adjusted_probs = np.zeros(len(initial_probs))
    for i, (idx, conf, cls) in enumerate(zip(all_sample_indices, all_sample_confidences, all_sample_labels)):
        # 当前样本的概率除以该类别在最终标签中的样本数
        if final_class_counts[cls] > 0:
            adjusted_probs[i] = initial_probs[i] / final_class_counts[cls]
        else:
            adjusted_probs[i] = initial_probs[i]

    # 重新归一化
    if np.sum(adjusted_probs) > 0:
        final_probs = adjusted_probs / np.sum(adjusted_probs)
    else:
        final_probs = np.ones(len(adjusted_probs)) / len(adjusted_probs)

    # 4. 直接按照概率对所有样本进行采样，直到达到采样总数
    final_samples = []
    final_labels = []

    # 采样逻辑
    if len(all_sample_indices) >= balance_sample_total:
        # 样本充足：按概率采样，无放回
        selected_indices = np.random.choice(
            range(len(all_sample_indices)),
            size=balance_sample_total,
            replace=False,
            p=final_probs
        )
    else:
        # 样本不足：按概率采样，有放回
        if not balance_replace:
            print(f"警告：总样本不足（{len(all_sample_indices)} 个），自动启用有放回采样")
        selected_indices = np.random.choice(
            range(len(all_sample_indices)),
            size=balance_sample_total,
            replace=True,
            p=final_probs
        )

    # 收集结果
    for selected_idx in selected_indices:
        final_samples.append(all_sample_indices[selected_idx])
        final_labels.append(all_sample_labels[selected_idx])

    # --------------------------
    # Step 5. 打乱样本顺序
    # --------------------------
    # if seed is not None:
    #     np.random.seed(seed)  # 固定种子确保可复现
    # shuffle_indices = np.random.permutation(len(final_samples))
    # final_samples = [final_samples[i] for i in shuffle_indices]
    # final_labels = [final_labels[i] for i in shuffle_indices]

    # --------------------------
    # Step 6. 输出采样结果统计
    # --------------------------
    class_counts = defaultdict(int)
    for label in final_labels:
        class_counts[label] += 1

    print("\n=== 最终采样结果 ===")
    for cls in sorted(class_counts.keys()):
        print(f"类别 {cls + 1}：{class_counts[cls]} 个样本")

    return final_samples, final_labels