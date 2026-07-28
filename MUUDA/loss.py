import torch
import torch.nn as nn

def compute_category_consistency_loss(
        band_weights,  # [B, C]
        outputs,  # [B, K] logits
        labels,
        eps=1e-6
):

    # 取真实类别 logit
    z = outputs.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]

    # 中心化（去尺度）
    z = z - z.mean()
    w = band_weights - band_weights.mean(dim=0, keepdim=True)

    # band-wise covariance with class evidence
    cov = (w * z.unsqueeze(1)).mean(dim=0)  # [C]

    # 归一化为相关系数（保证数值稳定 & 非负）
    var_w = w.var(dim=0) + eps
    var_z = z.var() + eps

    corr = cov.pow(2) / (var_w * var_z)  # [C]  ≥ 0

    # loss：鼓励 band 与类别判别“有关”
    loss = 1.0 - corr.mean()

    return loss


class CategoryConsistencyLoss(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(CategoryConsistencyLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        # 使用更小的标准差初始化，减少初始距离
        self.weightcenters = nn.Parameter(torch.zeros(num_classes, embedding_size))
        nn.init.normal_(self.weightcenters, mean=0, std=0.1)  # 减小标准差

    def forward(self, x, labels):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)

        # 确保weightcenters与x在同一设备上
        weightcenters = self.weightcenters.to(x.device)

        # 归一化输入特征，使距离计算更稳定
        x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)
        centers_normalized = torch.nn.functional.normalize(weightcenters, p=2, dim=1)

        # 使用余弦距离而不是欧几里得距离，数值范围在[0,2]之间
        # 1 - 余弦相似度 = 余弦距离
        centers = centers_normalized[labels]  # [batch_size, embedding_size]
        cosine_similarity = torch.sum(x_normalized * centers, dim=1)  # [batch_size]
        cosine_distance = 1.0 - cosine_similarity  # 范围[0,2]

        # 计算平均余弦距离作为损失
        loss = torch.mean(cosine_distance)

        return loss * 0.1

import torch.nn.functional as F


class ConsistencyLossTracker(nn.Module):
    """
    独立的类原型追踪器，避免全局状态污染
    
    功能：
    1. 类内一致性：强制相同类别的 band_w 趋向类中心
    2. 类间差异性：强制不同类别的 band_w 中心尽可能正交
    """
    def __init__(self, num_classes, band_dim, momentum=0.9, eps=1e-8):
        super(ConsistencyLossTracker, self).__init__()
        self.num_classes = num_classes
        self.band_dim = band_dim
        self.momentum = momentum
        self.eps = eps
        
        # 类原型存储
        self.w_proto = nn.Parameter(torch.zeros(num_classes, band_dim), requires_grad=False)
        self.initialized = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
    
    def forward(self, band_w, label):
        dev = band_w.device
        B, Band = band_w.shape
        
        # 确保在正确设备上
        band_w = band_w.to(dev)
        label = label.to(dev)
        
        # 1. 计算并更新类权重中心
        unique_labels = torch.unique(label)
        intra_loss = torch.tensor(0.0, device=dev)
        
        for cls in unique_labels:
            idx = int(cls.item())
            if idx >= self.num_classes:
                continue  # 跳过超出范围的类别
            
            mask = (label == cls)
            cls_band_w = band_w[mask]
            
            # 计算当前batch该类的权重均值
            cls_center_batch = cls_band_w.mean(dim=0)
            
            # 使用EMA更新全局原型
            if self.initialized[idx] == 0:
                self.w_proto[idx] = cls_center_batch.detach()
                self.initialized[idx] = 1
            else:
                self.w_proto[idx] = self.momentum * self.w_proto[idx] + (1 - self.momentum) * cls_center_batch.detach()
            
            # 类内一致性损失
            sample_to_proto_sim = F.cosine_similarity(cls_band_w, self.w_proto[idx].unsqueeze(0), dim=1)
            intra_loss += (1.0 - sample_to_proto_sim.mean())
        
        intra_loss = intra_loss / len(unique_labels)
        
        # 2. 类间差异性损失
        active_indices = torch.where(self.initialized > 0)[0]
        if len(active_indices) > 1:
            active_protos = self.w_proto[active_indices]
            active_protos_norm = F.normalize(active_protos, p=2, dim=1)
            inter_corr = torch.matmul(active_protos_norm, active_protos_norm.T)
            
            n_active = len(active_indices)
            identity = torch.eye(n_active, device=dev)
            inter_loss = ((inter_corr - identity) ** 2).sum() / (n_active * (n_active - 1))
        else:
            inter_loss = torch.tensor(0.0, device=dev)
        
        return intra_loss + inter_loss
    
    def reset(self):
        """重置原型状态，用于新实验"""
        self.w_proto.zero_()
        self.initialized.zero_()


def consistency_loss(band_w, img, label, tracker=None, momentum=0.9, eps=1e-8):
    """
    包装函数：支持使用独立tracker或创建临时tracker
    
    参数：
        tracker: ConsistencyLossTracker实例，如果为None则创建临时tracker（不推荐）
    """
    if tracker is None:
        # 创建临时tracker（每次调用都会重新初始化，不推荐用于训练）
        B, Band = band_w.shape
        current_max_cls = int(label.max().item())
        tracker = ConsistencyLossTracker(current_max_cls + 1, Band, momentum, eps)
    
    return tracker(band_w, label)
    