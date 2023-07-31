# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import numpy as np
import skimage
import skimage.filters
import torch.nn.functional as F
import scipy.ndimage as nd
import cv2
from ..DRD import DenseRelationDistill


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)  # out_channels=256�?        

        self.dense_relation = cfg.MODEL.DENSE_RELATION  # True
        self.dense_sum = cfg.MODEL.DENSE_SUM  # True

        self.layer3 = cfg.MODEL.LAYER3  # False
        self.layer1 = cfg.MODEL.LAYER1  # False
        self.layer0 = cfg.MODEL.LAYER0  # False
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        if self.dense_relation:  # True
            self.drd_opt = DenseRelationDistill(256, 32, 128, self.dense_sum)

        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16384, 1024)
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.RPN.FREEZE:
            for p in self.rpn.parameters():
                p.requires_grad = False
            print('froze rpn parameters')

        if cfg.MODEL.ROI_BOX_HEAD.FREEZE_FEAT:
            for p in self.roi_heads.box.feature_extractor.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

    def meta_extractor(self, meta_data, dr=False):
        # 决定使用RPN的第几个layer的特征。backbone返回的是一个列表，�?-3索引分别对应�?-3主层的输出�?        
        if self.layer3:
            base_feat = self.backbone((meta_data, 1))[3]
        elif self.layer1:
            base_feat = self.backbone((meta_data, 1))[1]
        elif self.layer0:
            base_feat = self.backbone((meta_data, 1))[0]
        else:
            # type(meta_data)=torch.Tensor
            # 这两句不加，单卡跑会报错
            device = torch.device("cuda")
            meta_data = meta_data.to(device)
            base_feat = self.backbone((meta_data, 1))[2]
        if dr:  # True
            return base_feat
        # 对base feat �?*2窗口的最大池化，然后调整尺寸，应该是会变成一�?6384的向量，进fc，变�?024维，接sigmoid
        feature = self.fc1(self.maxpool(base_feat).view(base_feat.shape[0], -1))
        feature = self.sigmoid(feature)

        return feature

    def forward(self, images, targets=None, meta_input=None, meta_attentions=None, average_shot=False, iteration=0):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # torch.cuda.empty_cache()
        # print(meta_input)
        # attentions是sup的特征，仅提取sup特征就返回，是最后保存类平均特征的部分�?        
        if average_shot:
            attentions = self.meta_extractor(meta_input, dr=self.dense_relation)
            return attentions
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            # 训练进入下一�?
        if self.training and meta_input is not None:
            attentions = self.meta_extractor(meta_input, dr=self.dense_relation)
        # 测试进入下一句，传入保存好的平均特征进来
        if meta_attentions is not None:
            attentions = meta_attentions

        images = to_image_list(images)
        # 经过backbone的query基础特征
        features = self.backbone((images.tensors, 0))  # =tuple len=5 why??应该�?个尺度的特征  len=5 这里�?和meta�?区别�?        
        if self.dense_relation:  # 使用DRD模块，利用图像特征和注意�?            
            features = self.drd_opt(features, attentions)  # attentions就是meta数据经过backbone的base feature

        proposals, proposal_losses = self.rpn(images, features, targets)

        # 下面是原MRCN�?        

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, iteration=iteration)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        # torch.cuda.empty_cache()
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
