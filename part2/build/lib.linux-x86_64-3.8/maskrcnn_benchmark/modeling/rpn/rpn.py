# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor

# 似乎没用
class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg

# 这个没用上
class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


# 将所有类别的attentions整合成一个
# 1. 先将每一类的2048维的attention向量->2048维
# 2. 将所有降维后的向量concat(如果直接concat起来，那维度就是动态变化的了)，直接相加算了
# 3. 将concat以后的向量->1024维
class AttentionsMerge(nn.Module):

    def __init__(self):
        super(AttentionsMerge, self).__init__()
        self.merge = nn.Linear(256, 128)  # 第三步
        self.conv1t = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 3*3卷积
        self.conv2t = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.conv3t = nn.Conv2d(256, 256, kernel_size=3, stride=3, padding=1)
        self.gap1t = nn.AdaptiveAvgPool2d((1, 1))
        for l in [self.conv1t, self.conv2t]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        self.convall=nn.Sequential(self.conv1t, self.conv2t, self.gap1t)


    def forward(self, attentions):
        device=attentions.device
        tsum=torch.sum(attentions, dim=0)
        tsum=tsum.view(1, tsum.size(0), tsum.size(1), tsum.size(2))
        vector=self.convall(tsum)
        vector=vector.view(vector.size(1))
        vector=F.softmax(vector,dim=0)
        return vector


# RPN 特征预测参数的部分
class Attention2Weights(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(Attention2Weights, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kerner_size = kernel_size

        self.metablock = nn.Sequential(
            nn.Linear(self.in_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_channel * 256 * kernel_size * kernel_size)
        )

    def forward(self, proto):
        weight = self.metablock(proto)  # 输入黄色向量，3个FC得到1*1 filters即weights

        return weight


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        mid_channels = int(in_channels / 4)
        self.num_anchors = num_anchors
        self.mid_channels = mid_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1) # 3*3卷积
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * mid_channels, kernel_size=1, stride=1) # 1*1卷积
        self.cls_logits = nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1)   # 1*1
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1) # 1*1

        # 新加
        self.Merge_Attention = AttentionsMerge()
        self.Class_Attention_Conv = Attention2Weights(in_channel=256, out_channel=self.num_anchors*self.mid_channels, kernel_size=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x, attentions=None, training=True):
        # 新加
        # attentions.shape=[nclass, 256, 16, 16]
        if isinstance(attentions, dict):
            device=x[0].device
            newattentions=None
            for i in attentions.values():
                if newattentions==None:
                    newattentions=i.view(1, i.size(0), i.size(1), i.size(2))
                else:
                    newattentions=torch.cat((newattentions, i.view(1, i.size(0), i.size(1), i.size(2))), dim=0)
            attentions=newattentions.to(device)
        weights=None
        if attentions is not None:
            merged_attention=self.Merge_Attention(attentions)   # [256]
            weights = self.Class_Attention_Conv(merged_attention).view(self.num_anchors*self.mid_channels, 256, 1, 1)


        features = []
        logits = []
        bbox_reg = []
        for feature in x:
            # feature.shape=[bs/GPU, 256, 13/25/50/100/200, 13/25/50/100/200]
            t = F.relu(self.conv(feature))
            cls_conv = self.cls_conv(t)

            # 新加，PCNN二维版尝试
            if weights is not None:
                atten_score = nn.functional.conv2d(t, weights)
                cls_conv=cls_conv*atten_score+cls_conv

            cls_conv = cls_conv.view(cls_conv.size(0), self.num_anchors, self.mid_channels, cls_conv.size(2), cls_conv.size(3))
            features.append(cls_conv)   # batch x num_anchor x midchannel x h x w
            cls_conv = cls_conv.view(cls_conv.size(0) * self.num_anchors, self.mid_channels, cls_conv.size(3), cls_conv.size(4))
            cls_logits = self.cls_logits(cls_conv).view(-1, self.num_anchors, cls_conv.size(2), cls_conv.size(3))



            logits.append(cls_logits)
            bbox_reg.append(self.bbox_pred(t))
            
        return logits, bbox_reg, features



class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]   # default "SingleConvRPNHead"
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, attentions=None, training=True):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression, rpn_features = self.head(features, attentions=attentions)
        
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:   # 进这里
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
