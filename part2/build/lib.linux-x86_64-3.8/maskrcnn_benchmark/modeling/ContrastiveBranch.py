import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):   # 1024 128
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                # 用XavierFill方法初始化权重，此时bias会初始化为0
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        # shape[0]应该是特征的个数吧，看上面的注释，[1]是特征的维数
        # features.shape [256*bs/GPU, 128], labels.shape [256*bs/GPU], ious.shape [256*bs/GPU]
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        # 如果输入labels是个向量（一维列表），则扩展为列向量，即每个特征对应的标签
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False，shape[256*bs/GPU, 256*bs/GPU]
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        # features和 features的转置 相乘后，逐元素除以temperature，对应文中zi*zj/τ，z是已经正则化过的，即计算zi和zj的cos相似度
        # similarity的尺寸是M*M，也就是M个特征之间的相似度，一个相似度矩阵 shape[256*bs/GPU, 256*bs/GPU]
        similarity = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        # dim=1找每行的最大值，即每个特征的最大值，因为keepdim所以返回的是[[8],[6]]这样的结构，M*1
        # 很明显，这里每行最大值应该是1，也就是自身和自身的相似度 shape [256*bs/GPU, 1]
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        # detach似乎会让requires_grad变成False，但这里意义不明
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        # 和similarity相同尺寸的全1矩阵，shape[256*bs/GPU, 256*bs/GPU]
        logits_mask = torch.ones_like(similarity)
        # 对角线值填充为0
        logits_mask.fill_diagonal_(0)

        # shape[256*bs/GPU, 256*bs/GPU]
        exp_sim = torch.exp(similarity) * logits_mask

        # shape[256*bs/GPU, 256*bs/GPU]
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # shape[256*bs/GPU]
        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        # keep为满足iou阈值的索引，对应论文中f(ui)里的第一项，即选出满足iou阈值的，shape [256*bs/GPU]
        keep = ious >= self.iou_threshold
        # shape 5-11 变化 决定了loss和coef的shape，其shape和torch.sum(keep)相同，也就是提取出iou大于阈值的几项
        per_label_log_prob = per_label_log_prob[keep]
        # shape 5-11 变化
        loss = -per_label_log_prob

        # 一般是none，即torch.ones_like(iou) 返回一个填充了标量值1的张量，其大小与input相同。
        # 对应论文f(ui)里的第二项，即reweighting函数g(ui)，给不同iou得分的proposal赋不同的权重
        # 根据论文消融，最好的权重似乎就是1，可以认为是没做什么
        # coef的shape和loss相同，变化的
        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay
