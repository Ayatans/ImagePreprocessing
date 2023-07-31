# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# maskrcnn自带，已废弃，因为本实验仅用GeneralizedRCNN，无需
from .generalized_rcnn import GeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]  # 仅有一个上面的GRCNN
    return GeneralizedRCNN(cfg)   # 所以返回的肯定是GeneralizedRCNN(cfg)
