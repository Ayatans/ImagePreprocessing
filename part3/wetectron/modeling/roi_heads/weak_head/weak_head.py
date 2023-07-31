# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch import nn
import numpy as np

from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from ..box_head.loss import make_roi_box_loss_evaluator
from ..box_head.roi_box_predictors import make_roi_box_predictor
from ..box_head.inference import make_roi_box_post_processor as strong_roi_box_post_processor

from .roi_weak_predictors import make_roi_weak_predictor
from .inference import make_roi_box_post_processor as weak_roi_box_post_processor
from .loss import make_roi_weak_loss_evaluator, generate_img_label
from .roi_sampler import make_roi_sampler

from wetectron.modeling.utils import cat
from wetectron.structures.boxlist_ops import cat_boxlist
from wetectron.modeling.roi_heads.sim_head.sim_net import Sim_Net

# 没有回归分支，不能用！
class ROIWeakHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIWeakHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = weak_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_weak_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, model_cdb=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        cls_score, det_score, ref_scores = self.predictor(x, proposals)
        if not self.training:
            if ref_scores == None:
                final_score = cls_score * det_score
            else:
                final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.post_processor(final_score, proposals)
            return x, result, {}, {}

        loss_img, accuracy_img = self.loss_evaluator([cls_score], [det_score], ref_scores, proposals, targets)

        return (
            x,
            proposals,
            loss_img,
            accuracy_img
        )


class ROIWeakRegHead(torch.nn.Module):
    """ Generic Box Head class w/ regression. """
    def __init__(self, cfg, in_channels):
        super(ROIWeakRegHead, self).__init__()
        # VGG16.roi_head in_channels必须=512
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        # "MISTPredictor" 是对比方法的东西
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        # RoIRegLoss
        self.loss_evaluator = make_roi_weak_loss_evaluator(cfg)
        # 没用weak用的strong
        self.weak_post_processor = weak_roi_box_post_processor(cfg)
        self.strong_post_processor = strong_roi_box_post_processor(cfg)

        # "AVG"
        self.HEUR = cfg.MODEL.ROI_WEAK_HEAD.REGRESS_HEUR

        # None
        self.roi_sampler = make_roi_sampler(cfg) if cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS != "none" else None

        # 'dropblock'
        self.DB_METHOD = cfg.DB.METHOD

        # 求相似度的，FC+Relu+FC+Norm，论文4.1，最终输出的维度128直接在里面定义了不在config
        self.model_sim = Sim_Net(cfg, self.feature_extractor.out_channels)

    def go_through_cdb(self, features, proposals, model_cdb):
        # 其实不进这个if。测试时+method=none 就提取特征，default是none，yaml是dropblock。
        if not self.training or self.DB_METHOD == "none":
            return self.feature_extractor(features, proposals)
        elif self.DB_METHOD == "concrete":
            pooled_feats = self.feature_extractor.forward_pooler(features, proposals)
            x = model_cdb(pooled_feats)
            return self.feature_extractor.forward_neck(x), pooled_feats
        # 进这里
        elif self.DB_METHOD == "dropblock":
            return self.feature_extractor.forward_dropblock(features, proposals)
        elif self.DB_METHOD == "attention":
            return self.feature_extractor.forward_attention_dropblock(features, proposals)
        else:
            raise ValueError

    def forward(self, features, proposals, targets=None, model_cdb=None, iteration=None):
        # for partial labels
        # none 不进这个if
        if self.roi_sampler is not None and self.training:
            with torch.no_grad():
                proposals = self.roi_sampler(proposals, targets)

        # VGG16.roi_head.forward 提取proposal 特征，
        # 输出roi feats是4096维向量，所以这里对应论文里的yita，生成的是用于MILhead的2分支的v
        # 输出pooled feats是RoI特征，对应文中图里上面的f，尺寸可能是7*7*512
        clean_roi_feats, clean_pooled_feats = self.feature_extractor.forward(features, proposals)

        # 这一串操作对应图2a前部分的dropblock和η函数
        if self.training:
            # 先对扩增前的proposal features计算相似度，论文里的φ函数。为什么是扩增前的？
            # 输出sim feature是Fig.2(b)的S 4.1节最后一句话说了，相似度特征是dropblock前的RoI特征向量算出来的。
            sim_feature = self.model_sim(clean_roi_feats)
            # =feature_extractor.forward_dropblock(features, proposals)，注意dropblock的参数是直接在init里改的，不是cfg
            # 输出是f~
            aug_pooled_feats = self.go_through_cdb(clean_pooled_feats, proposals, model_cdb=model_cdb)
            # 特征展平，然后过 FC-ReLU-Dropout-FC-ReLU-Dropout，变成论文图2上的vn
            aug_roi_feats = self.feature_extractor.forward_neck(aug_pooled_feats)
            # MIDN 基础分类器啊 图里第一行检测器 返回：WSDDN分类得分、检测得分，OICR各级分类得分、回归得分的列表，回归分支和分类分支结构相同都是1个FC
            # 训练时此时没有进行softmax，是在loss evaluator里进行的
            cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor(aug_roi_feats, proposals)

        # 测试时：直接检测，在self.predictor里进行softmax了
        if not self.training:
            cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor(clean_roi_feats, proposals)
            result = self.testing_forward(cls_score, det_score, proposals, ref_scores, ref_bbox_preds)
            return clean_roi_feats, result, {}, {}

        loss_img, accuracy_img = self.loss_evaluator([cls_score], [det_score], ref_scores, ref_bbox_preds, sim_feature, clean_pooled_feats, self.feature_extractor, self.model_sim, proposals, targets)

        return aug_roi_feats, proposals, loss_img, accuracy_img

    def testing_forward(self, cls_score, det_score, proposals, ref_scores=None, ref_bbox_preds=None):
        if self.HEUR == "WSDDN":
            final_score = cls_score * det_score
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "CLS-AVG":
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.weak_post_processor(final_score, proposals)
        # 用的这个，把各个得分叠在一起求平均
        elif self.HEUR == "AVG":
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            final_regression = torch.mean(torch.stack(ref_bbox_preds), dim=0)
            result = self.strong_post_processor((final_score, final_regression), proposals, softmax_on=False)
        elif self.HEUR == "UNION":
            prop_list = [len(p) for p in proposals]
            ref_score_list = [rs.split(prop_list) for rs in ref_scores]
            ref_bbox_list = [rb.split(prop_list) for rb in ref_bbox_preds]
            final_score = [torch.cat((ref_score_list[0][i], ref_score_list[1][i], ref_score_list[2][i])) for i in range(len(proposals)) ]
            final_regression = [torch.cat((ref_bbox_list[0][i], ref_bbox_list[1][i], ref_bbox_list[2][i])) for i in range(len(proposals)) ]
            augmented_proposals = [cat_boxlist([p for _ in range(3)]) for p in proposals]
            result = self.strong_post_processor((cat(final_score), cat(final_regression)), augmented_proposals, softmax_on=False)
        else:
            raise ValueError
        return result


def build_roi_weak_head(cfg, in_channels):
    """
    Constructs a new weak head.
    By default, uses ROIWeakRegHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    # True 进这里
    if cfg.MODEL.ROI_WEAK_HEAD.REGRESS_ON:
        return ROIWeakRegHead(cfg, in_channels)
    else:
        return ROIWeakHead(cfg, in_channels)
