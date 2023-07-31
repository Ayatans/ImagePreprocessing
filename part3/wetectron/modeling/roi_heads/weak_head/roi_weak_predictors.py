# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F

from wetectron.modeling import registry


@registry.ROI_WEAK_PREDICTOR.register("WSDDNPredictor")
class WSDDNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(WSDDNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)

        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
        else:
            final_det_logit = det_logit

        return cls_logit, final_det_logit, None


@registry.ROI_WEAK_PREDICTOR.register("OICRPredictor")
class OICRPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(OICRPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)

        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.ref3 = nn.Linear(num_inputs, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)
        ref1_logit = self.ref1(x)
        ref2_logit = self.ref2(x)
        ref3_logit = self.ref3(x)

        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            #
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit

        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        return cls_logit, final_det_logit, ref_logits


# 都用的这个
@registry.ROI_WEAK_PREDICTOR.register("MISTPredictor")
class MISTPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MISTPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES # 类别数
        # False, num_bbox_reg_classes=num_classes
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)

        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred1 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred2 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref3 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred3 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref4 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred4 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        #if not self.training:
            #torch.nn.Parameter(torch.where(torch.ge(self.ref3.weight, 0.0), self.ref3.weight, torch.zeros_like(self.ref3.weight)))
            #self.ref1.weight = torch.nn.Parameter(torch.cat((self.ref1.weight[0].view(1,-1), torch.where(torch.ge(self.ref1.weight[1:], 0.0), self.ref1.weight[1:], torch.zeros_like(self.ref1.weight[1:])))))
            #self.ref2.weight = torch.nn.Parameter(torch.cat((self.ref2.weight[0].view(1,-1), torch.where(torch.ge(self.ref2.weight[1:], 0.0), self.ref2.weight[1:], torch.zeros_like(self.ref2.weight[1:])))))
            #self.ref3.weight = torch.nn.Parameter(torch.cat((self.ref3.weight[0].view(1,-1), torch.where(torch.ge(self.ref3.weight[1:], 0.0), self.ref3.weight[1:], torch.zeros_like(self.ref3.weight[1:])))))

        #    self.bbox_pred1.weight = torch.nn.Parameter(torch.cat((self.bbox_pred1.weight[:21], torch.where(torch.ge(self.bbox_pred1.weight[21:], 0.0), self.bbox_pred1.weight[21:], torch.zeros_like(self.bbox_pred1.weight[21:])))))
        #    self.bbox_pred2.weight = torch.nn.Parameter(torch.cat((self.bbox_pred2.weight[:21], torch.where(torch.ge(self.bbox_pred2.weight[21:], 0.0), self.bbox_pred2.weight[21:], torch.zeros_like(self.bbox_pred2.weight[21:])))))
        #    self.bbox_pred3.weight = torch.nn.Parameter(torch.cat((self.bbox_pred3.weight[:21], torch.where(torch.ge(self.bbox_pred3.weight[21:], 0.0), self.bbox_pred3.weight[21:], torch.zeros_like(self.bbox_pred3.weight[21:])))))

        cls_logit = self.cls_score(x)   # WSDDN的分类得分
        det_logit = self.det_score(x)   # WSDDN的检测得分
        ref1_logit = self.ref1(x)       # OICR stage 1 分类分支
        bbox_pred1 = self.bbox_pred1(x) # OICR stage 1 回归分支 下同 有几个就是K=几
        ref2_logit = self.ref2(x)
        bbox_pred2 = self.bbox_pred2(x)
        ref3_logit = self.ref3(x)
        bbox_pred3 = self.bbox_pred3(x)
        # ref4_logit = self.ref4(x)
        # bbox_pred4 = self.bbox_pred4(x)

        # 测试时
        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs # 切分成batch size个 也就是对每个图做操作
            det_logit_list = det_logit.split([len(p) for p in proposals])
            # 逐图操作完再concat起来
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0) # 按行
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)

            ref1_logit = F.softmax(ref1_logit, dim=1)   # 分类得分和上面一样，按列softmax
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
            # ref4_logit = F.softmax(ref4_logit, dim=1)
        else:
            # 为什么训练时不用softmax？？？
            final_det_logit = det_logit


        '''可用于消融refinement head 的stage'''
        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]
        # ref_logits = [ref1_logit, ref2_logit]
        # bbox_preds = [bbox_pred1, bbox_pred2]

        return cls_logit, final_det_logit, ref_logits, bbox_preds

    # 没用
    def forward_final(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)

        cls_logit = F.softmax(cls_logit, dim=1)
        det_logit = F.softmax(det_logit, dim=0)
        import IPython; IPython.embed()
        # do softmax along ROI for different imgs
        #det_logit_list = det_logit.split([len(p) for p in proposals])
        #final_det_logit = []
        #for det_logit_per_image in det_logit_list:
        #    det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
        #    final_det_logit.append(det_logit_per_image)
        #final_det_logit = torch.cat(final_det_logit, dim=0)
        #import IPython; IPython.embed()

    # 没用
    def forward_ref(self, x):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        ref1_logit = self.ref1(x)
        ref2_logit = self.ref2(x)
        ref3_logit = self.ref3(x)

        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        return ref_logits

def make_roi_weak_predictor(cfg, in_channels):
    func = registry.ROI_WEAK_PREDICTOR[cfg.MODEL.ROI_WEAK_HEAD.PREDICTOR]   # default和yaml都是"MISTPredictor"
    return func(cfg, in_channels)
