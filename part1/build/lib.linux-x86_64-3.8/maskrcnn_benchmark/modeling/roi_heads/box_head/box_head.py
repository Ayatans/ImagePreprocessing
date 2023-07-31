# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from ....structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg
import random
import time
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, attentions=None, meta_label=None, iteration=0):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        xc, xr = self.feature_extractor(features, proposals)
        
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
                if label[i] == 0: continue
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
            f.savefig(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+'.jpg', dpi=1000)
            f.clear()
            
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(xc, xr)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return xc, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression])      
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
    return ROIBoxHead(cfg, in_channels)
