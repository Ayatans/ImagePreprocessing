# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-
import torch

from .box_head.box_head import build_roi_box_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()


    def forward(self, features, proposals, iteration, targets=None, attentions=None, meta_label=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        '''
        self.box 是ROIBoxHead类 
        ROIBoxHead(
          (feature_extractor): FPN2MLPFeatureExtractor(
            (pooler): Pooler(
              (poolers): ModuleList(
                (0): ROIAlign(output_size=(8, 8), spatial_scale=0.25, sampling_ratio=2)
                (1): ROIAlign(output_size=(8, 8), spatial_scale=0.125, sampling_ratio=2)
                (2): ROIAlign(output_size=(8, 8), spatial_scale=0.0625, sampling_ratio=2)
                (3): ROIAlign(output_size=(8, 8), spatial_scale=0.03125, sampling_ratio=2)
              )
            )
            (avgpooler): AdaptiveAvgPool2d(output_size=(8, 8))
            (fc6c): Linear(in_features=16384, out_features=1024, bias=True)
            (fc7c): Linear(in_features=1024, out_features=1024, bias=True)
            (fc6r): Linear(in_features=16384, out_features=1024, bias=True)
            (fc7r): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (predictor): FPNPredictor(
            (cls_score): Linear(in_features=1024, out_features=21, bias=True)
            (bbox_pred): Linear(in_features=1024, out_features=4, bias=True)
          )
          (post_processor): PostProcessor()
        )
        '''
        x, detections, loss_box = self.box(features, proposals, iteration, targets, attentions, meta_label)
        losses.update(loss_box)


        return x, detections, losses

class CLCombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CLCombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()


    def forward(self, features, proposals, targets=None, attentions=None, meta_label=None):
        losses_box = {}
        losses_contra={}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        # 这里是不是可以把对比分支和原来的两个拆开来啊？毕竟都combinedheads了
        x, detections, loss_box, loss_contra = self.box(features, proposals, targets, attentions, meta_label)
        losses_box.update(loss_box)
        losses_contra.update(loss_contra)

        return x, detections, losses_box,losses_contra


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together afterwards
    roi_heads = []
    # False 不走这里
    if cfg.MODEL.RETINANET_ON:
        return []

    # False 走这里
    if not cfg.MODEL.RPN_ONLY:
        # 要么ROIBoxHead，要么CLROIBoxHead。这里的"box"就是上面调用该ROIBoxHead时的self.box的来源
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        # if cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_CONTRASTIVE_BRANCH:
        #     roi_heads = CLCombinedROIHeads(cfg, roi_heads)
        # else:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
