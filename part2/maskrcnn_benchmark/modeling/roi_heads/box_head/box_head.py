# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from ....structures.boxlist_ops import boxlist_iou
from ...ContrastiveBranch import ContrastiveHead
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg
import random
import time
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.bbox, boxes2.bbox

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)  # 这个应该不用改，参考Fig3 没变
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        # 新加
        '''ROI box head 里的FC层的隐藏层的维度 default 1024'''
        self.fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        '''7352页左侧Dc值，对比头里的1层MLP。default默认128，tau的config里是256，t代表temperature'''

        if cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_CONTRASTIVE_BRANCH:
            self.mlp_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
            self.contra_encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)

    def forward(self, features, proposals, iteration, targets=None, attentions=None, meta_label=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels, len=5
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.len=bs/GPU，每项的bbox shape=gt个数*4

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList], len=4): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # 新加
        # 此时proposals每项有2k+个bbox，imgsize,fields只有objectness。在这里要给proposals加上iou等fields！仿照FSCE

        # for thisproposal, thistarget in zip(proposals, targets):
        #     match_quality_matrix=boxlist_iou(thistarget, thisproposal)
        #     iou, _ = match_quality_matrix.max(dim=0)
        #     # matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)





        # 下面这个减采样会导致proposal[0]的len减少，低于256的不变，高于256的最高是256
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            '''
            proposals每项是BoxList，extra_fields的键有['objectness', 'labels', 'regression_targets', 'matched_idxs']
            每个BoxList还有bbox属性，shape=[256*bs/GPU, 4]，即每张图的256个proposal的box
            objectness:每个proposal的前景目标得分，shape=[256*bs/GPU]
            labels:每个proposal对应的标签，shape=[256*bs/GPU]
            regression_targets:每个proposal 框的偏移量 shape [256*bs/GPU, 4]
            matched_idxs:很多-1，其余值到几十，没弄清意义，shape=[256*bs/GPU]
            '''
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        # xc和xr shape相同，均为[ *, 1024]，每张卡多图时*会变动；一卡一图时固定[256, 1024]
        # xc是用于分类得分，xr用于box回归，就是proposal的特征  xr比xc最后多relu了一次
        xc, xr = self.feature_extractor(features, proposals)    # 里面先池化再提proposal features

        # 此时，proposals每项有256个框，fields：['objectness', 'labels', 'regression_targets', 'matched_idxs']

        def plot_embedding(data, label, title, show=None):
            # param data:data
            # param label:label
            # param title:title of output
            # param show:(int) if you have too much proposals to draw, you can draw part of them
            # return: tsne-image

            if show is not None:
                temp = [i for i in range(len(data))]
                random.shuffle(temp)
                data = data[temp]
                data = data[:show]
                label = torch.tensor(label)[temp]
                label = label[:show]
                label.numpy().tolist()

            x_min, x_max = np.min(data, 0), np.max(data, 0)
            data = (data - x_min) / (x_max - x_min)  # norm data
            fig = plt.figure()

            # go through all the samples
            data = data.tolist()
            label = label.squeeze().tolist()
            for i in range(len(data)):
                if label[i]==0: continue
                plt.text(data[i][0], data[i][1], ".", fontsize=18, color=plt.cm.tab20(label[i] / 20))
            plt.title(title, fontsize=14)
            return fig

        # weight:(n proposals * 1024-D) input of the classifier
        # label: the label of the proposals/ground truth
        # we only select foreground proposals to visualize
        # you can try to visualize the weight of different classes by extracting weight during training or testing stage

        if self.training and iteration==cfg.SOLVER.MAX_ITER:
            print('making t-SNE')
            ts = TSNE(n_components=2, init='pca', random_state=0)
            weight = ts.fit_transform(xc.cpu().data.numpy())
            fig = plot_embedding(weight, cat([p.get_field('labels') for p in proposals], dim=0), 't-SNE feature')
            f=plt.gcf()
            f.savefig('tsne-img/'+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+'.jpg', dpi=1000)
            f.clear()

        # final classifier that converts the features into predictions, 输出2个Tensor
        # [256*bs/GPU, 21（类别数）], [256*bs/GPU,84]
        class_logits, box_regression = self.predictor(xc, xr)

        # contrastive encoding, shape [256, 128]
        # TODO: 每张卡多图时第一维可能不是256*bs/GPU了，后面研究下有影响没

        if cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_CONTRASTIVE_BRANCH:
            box_features_contrast=self.contra_encoder(xr)
            box_features_contrast=box_features_contrast.to(torch.device('cuda'))


        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return xc, result, {}
        if cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_CONTRASTIVE_BRANCH:
            loss_classifier, loss_box_reg, loss_contrastive = self.loss_evaluator(
                [class_logits], [box_regression], box_features_contrast=box_features_contrast, proposals=proposals, iteration=iteration)
            return (
                xc,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_contrastive=loss_contrastive),
            )
        else:
            loss_classifier, loss_box_reg = self.loss_evaluator(
                [class_logits], [box_regression], proposals=proposals, iteration=iteration)
            return (
                xc,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            )




def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    # if not cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_CONTRASTIVE_BRANCH:
    return ROIBoxHead(cfg, in_channels)
    # else:
    #     return CLROIBoxHead(cfg, in_channels)
