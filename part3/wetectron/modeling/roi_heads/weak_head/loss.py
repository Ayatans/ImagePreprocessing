# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
import collections
import torch.nn as nn
import random
import os
import numpy as np
from torch.nn import functional as F

from wetectron.layers import smooth_l1_loss
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms_index
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.matcher import Matcher
from wetectron.utils.utils import to_boxlist, cal_iou, easy_nms, cos_sim, get_share_class, generate_img_label
from .pseudo_label_generator import oicr_layer, mist_layer, od_layer
from wetectron.modeling.roi_heads.sim_head.sim_loss import Supcon_Loss, SupConLossV2
from wetectron.modeling.roi_heads.sim_head.sim_net import Sim_Net

def compute_avg_img_accuracy(labels_per_im, score_per_im, num_classes):
    """
       the accuracy of top-k prediction
       where the k is the number of gt classes
    """
    num_pos_cls = max(labels_per_im.sum().int().item(), 1)
    cls_preds = score_per_im.topk(num_pos_cls)[1]
    accuracy_img = labels_per_im[cls_preds].mean()
    return accuracy_img


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

@registry.ROI_WEAK_LOSS.register("WSDDNLoss")
class WSDDNLossComputation(object):
    """ Computes the loss for WSDDN."""
    def __init__(self, cfg):
        self.type = "WSDDN"

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-10):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
        Returns:
            img_loss (Tensor)
            accuracy_img (Tensor): the accuracy of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        total_loss = 0
        accuracy_img = 0
        for final_score_per_im, targets_per_im in zip(final_score_list, targets):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            total_loss += F.binary_cross_entropy(img_score_per_im, labels_per_im)
            accuracy_img += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)

        total_loss = total_loss / len(final_score_list)
        accuracy_img = accuracy_img / len(final_score_list)
        return dict(loss_img=total_loss), dict(accuracy_img=accuracy_img)


@registry.ROI_WEAK_LOSS.register("RoILoss")
class RoILossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.type = "RoI_loss"
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-8):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
            ref_scores
            proposals
            targets
        Returns:
            return_loss_dict (dictionary): all the losses
            return_acc_dict (dictionary): all the accuracies of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)
        for i in range(num_refs):
            return_loss_dict['loss_ref%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                lmda = 3 if i == 0 else 1
                pseudo_labels, loss_weights = self.roi_layer(proposals_per_image, source_score, labels_per_im, device)
                return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l, a in zip(return_loss_dict.keys(), return_acc_dict.keys()):
            return_loss_dict[l] /= len(final_score_list)
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict

#用这个
@registry.ROI_WEAK_LOSS.register("RoIRegLoss")
class RoIRegLossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        self.refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        # 测试default false，训练cfg true
        self.contra = cfg.SOLVER.CONTRA

        if self.refine_p > 0 and self.refine_p < 1 and not self.contra:
            self.mist_layer = mist_layer(self.refine_p)
        self.oicr_layer = oicr_layer()
        self.od_layer = od_layer()

        # for regression
        self.cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        # for partial labels， False
        self.roi_refine = cfg.MODEL.ROI_WEAK_HEAD.ROI_LOSS_REFINE
        self.partial_label = cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS
        assert self.partial_label in ['none', 'point', 'scribble']
        self.proposal_scribble_matcher = Matcher(
            0.5, 0.5, allow_low_quality_matches=False,
        )

        self.nms = cfg.nms
        self.sim_lmda = cfg.lmda
        self.pos_update = cfg.pos_update
        self.p_thres = cfg.thres
        self.p_iou = cfg.iou
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.temp = cfg.temp
        if cfg.loss == 'supcon':
            self.sim_loss = Supcon_Loss(self.temp)
        # 是v2这个
        elif cfg.loss == 'supconv2':
            self.sim_loss = SupConLossV2(self.temp)
        self.output_dir = cfg.OUTPUT_DIR


    # 没用
    def filter_pseudo_labels(self, pseudo_labels, proposal, target):
        """ refine pseudo labels according to partial labels """
        if 'scribble' in target.fields() and self.partial_label=='scribble':
            scribble = target.get_field('scribble')
            match_quality_matrix_async = boxlist_iou_async(scribble, proposal)
            _matched_idxs = self.proposal_scribble_matcher(match_quality_matrix_async)
            pseudo_labels[_matched_idxs < 0] = 0
            matched_idxs = _matched_idxs.clone().clamp(0)
            _labels = target.get_field('labels')[matched_idxs]
            pseudo_labels[pseudo_labels != _labels.long()] = 0

        elif 'click' in target.fields() and self.partial_label=='point':
            clicks = target.get_field('click').keypoints
            clicks_tiled = torch.unsqueeze(torch.cat((clicks, clicks), dim=1), dim=1)
            num_obj = clicks.shape[0]
            box_repeat = torch.cat([proposal.bbox.unsqueeze(0) for _ in range(num_obj)], dim=0)
            diff = clicks_tiled - box_repeat
            matched_ids = (diff[:,:,0] > 0) * (diff[:,:,1] > 0) * (diff[:,:,2] < 0) * (diff[:,:,3] < 0)
            matched_cls = matched_ids.float() * target.get_field('labels').view(-1, 1)
            pseudo_labels_repeat = torch.cat([pseudo_labels.unsqueeze(0) for _ in range(matched_ids.shape[0])])
            correct_idx = (matched_cls == pseudo_labels_repeat.float()).sum(0)
            pseudo_labels[correct_idx==0] = 0

        return pseudo_labels

    def __call__(self, class_score, det_score, ref_scores, ref_bbox_preds, sim_feature, clean_pooled_feats, feature_extractor, model_sim, proposals, targets, epsilon=1e-8):
        # WSDDN分类得分按列softmax，见论文
        class_score = F.softmax(cat(class_score, dim=0), dim=1)
        class_score_list = class_score.split([len(p) for p in proposals])

        # WSDDN检测得分先拆分成各图片的proposal，再按行softmax，见论文
        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])   # 按batch size拆分图片
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)
        detection_score_list = final_det_score.split([len(p) for p in proposals])

        # WSDDN proposal得分
        final_score = class_score * final_det_score
        # 按图拆分WSDDN proposal得分
        final_score_list = final_score.split([len(p) for p in proposals])

        device = class_score.device
        # C类
        num_classes = class_score.shape[1]

        # OICR各级的分类得分，进行softmax
        ref_score = ref_scores.copy()
        for r, r_score in enumerate(ref_scores):
            ref_score[r] = F.softmax(r_score, dim=1)
        avg_score = torch.stack(ref_score).mean(0).detach()
        avg_score_split = avg_score.split([len(p) for p in proposals])

        # 按图拆分OICR的分类得分和回归得分
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]
        ref_bbox_preds = [rbp.split([len(p) for p in proposals]) for rbp in ref_bbox_preds]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)  # stage数，K

        # 初始化各项loss
        for i in range(num_refs):
            return_loss_dict['loss_ref_cls%d'%i] = 0
            return_loss_dict['loss_ref_reg%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        pos_classes = [generate_img_label(num_classes, target.get_field('labels').unique(), device)[1:].eq(1).nonzero(as_tuple=False)[:,0] for target in targets]
        # 计算了对比损失
        if self.contra:
            return_loss_dict['loss_sim'] = 0
            # 按图拆分相似度特征向量和RoI特征
            sim_feature = sim_feature.split([len(p) for p in proposals])
            clean_pooled_feat = clean_pooled_feats.split([len(p) for p in proposals])

            pgt_index = [[torch.zeros((0), dtype=torch.long, device=device) for x in range(num_classes-1)] for y in range(len(targets))]
            pgt_collection = [torch.zeros((0), dtype=torch.float, device=device) for x in range(num_classes-1)]
            pgt_update = [torch.zeros((0), dtype=torch.float, device=device) for x in range(num_classes-1)]
            instance_diff = torch.zeros((0), dtype=torch.float, device=device)


            for idx, (final_score_per_im, pos_classes_per_im, proposals_per_image) in enumerate(zip(final_score_list, pos_classes, proposals)):
                # 按照各stage来
                for i in range(num_refs):
                    # 第1阶段的监督信息来自WSDDN的输出，其余的来自上一阶段
                    source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                    # 去掉背景的得分，仅考虑各类别的
                    proposal_score = source_score[:, 1:].clone()
                    # 对于每张图的标签所含有的类别
                    for pos_c in pos_classes_per_im:
                        # 找到最高得分的索引
                        max_index = torch.argmax(proposal_score[:,pos_c])
                        # 计算该图的所有proposal和该最高得分proposal的IoU
                        overlaps, _ = cal_iou(proposals_per_image, max_index, self.p_thres)
                        max_index=torch.tensor([max_index.item()], device=device)
                        '''IoU Sampling正常情况'''
                        # pgt_index[idx][pos_c] = torch.cat((pgt_index[idx][pos_c], overlaps)).unique()
                        '''消融 仅去掉 IoU Sampling'''
                        pgt_index[idx][pos_c] = torch.cat((pgt_index[idx][pos_c], max_index)).unique()


                for pos_c in pos_classes_per_im:
                    '''IoU Sampling'''
                    iou_samples = pgt_index[idx][pos_c]
                    pgt_update[pos_c] = torch.cat((pgt_update[pos_c], sim_feature[idx][iou_samples]))
                    hardness = final_score_list[idx][iou_samples, pos_c+1] / final_score_list[idx][:,pos_c+1].sum()
                    #hardness = avg_score_split[idx][iou_samples, pos_c+1] / avg_score_split[idx][:, pos_c+1].sum()
                    instance_diff = torch.cat((instance_diff, hardness))

                    '''Random Masking 其实就是一个Dropblock'''
                    # drop_logit = feature_extractor.forward_neck(feature_extractor.drop_pool(clean_pooled_feat[idx][iou_samples]))
                    # pgt_update[pos_c] = torch.cat((pgt_update[pos_c], model_sim(drop_logit) ))
                    # instance_diff = torch.cat((instance_diff, hardness))

                    '''高斯噪声'''
                    # noise_logit = feature_extractor.forward_neck(feature_extractor.noise_pool(clean_pooled_feat[idx][iou_samples]))
                    # pgt_update[pos_c] = torch.cat((pgt_update[pos_c], model_sim(noise_logit) ))
                    # instance_diff = torch.cat((instance_diff, hardness))

                    pgt_collection[pos_c] = pgt_update[pos_c].clone()

            # pgt=pseudo ground truth 图像数-阶段K-类别数
            pgt_instance = [[[torch.zeros((0), dtype=torch.long, device=device) for x in range(num_classes-1)] for z in range(num_refs)] for y in range(len(targets))]

            for idx, (final_score_per_im, pos_classes_per_im, proposals_per_image) in enumerate(zip(final_score_list, pos_classes, proposals)):
                for i in range(num_refs):
                    # 第1阶段的监督信息来自WSDDN的输出，其余的来自上一阶段
                    source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                    # 去掉背景的得分，仅考虑各类别的
                    proposal_score = source_score[:, 1:].clone()
                    # 对于每张图的标签所含有的类别
                    for pos_c in pos_classes_per_im:
                        # 找到最高得分的索引
                        max_index = torch.argmax(proposal_score[:,pos_c])

                        ''' 4.3 object discovery sim_thresh就是公式9里的τ'''
                        # 各相似度特征向量两两相乘
                        sim_mat = torch.mm(sim_feature[idx], sim_feature[idx].T)
                        '''公式9 求τ 某类最高分的相似度特征向量 * 其他同类proposal的相似度特征向量'''
                        sim_thresh = torch.mm(sim_feature[idx][max_index].view(1,-1), pgt_collection[pos_c].T).mean()

                        # 如果当前图像不止包含1种类别的目标
                        if pos_classes_per_im.shape[0] > 1:
                            # 对当前类而言的负类
                            neg_classes = pos_classes_per_im[(pos_classes_per_im != pos_c)]
                            '''公式10 得分最高的proposal和其他proposal相乘后（即相似度）大于τ的索引'''
                            sim_close = torch.ge(sim_mat[max_index], sim_thresh)    # 原sim thresh

                            '''4.3 object discovery sim thresh 消融'''
                            # sim_close=torch.ones(sim_mat[max_index].shape).to(sim_mat.device)

                            for neg_c in neg_classes:
                                # 负类中得分最高的proposal
                                neg_max_index = torch.argmax(proposal_score[:,neg_c])
                                sim_close = torch.ge(sim_close, sim_mat[neg_max_index])
                            sim_close = sim_close.nonzero(as_tuple=False).view(-1)
                        else:
                            sim_close = torch.ge(sim_mat[max_index], sim_thresh).nonzero(as_tuple=False).view(-1)

                        # nms 公式10下面的那句话
                        sim_close = easy_nms(proposals_per_image, sim_close, proposal_score[:,pos_c], nms_iou=self.nms)
                        # avoid none
                        sim_close = torch.cat((sim_close, max_index.view(-1))) if sim_close.nelement() == 0 else sim_close
                        # 添加进伪标签，公式10下面的那句话
                        pgt_instance[idx][i][pos_c] = torch.cat((pgt_instance[idx][i][pos_c], sim_close))

                        # ？？？
                        dup = torch.cat((sim_close, pgt_index[idx][pos_c])).unique()[torch.where(torch.cat((sim_close, pgt_index[idx][pos_c])).unique(return_counts=True)[1]>1)]
                        sim_close = torch.cat((sim_close,dup)).unique()[torch.where(torch.cat((sim_close,dup)).unique(return_counts=True)[1]==1)]
                        # avoid none
                        sim_close = torch.cat((sim_close, max_index.view(-1))) if sim_close.nelement() == 0 else sim_close

                        pgt_update[pos_c] = torch.cat((pgt_update[pos_c], sim_feature[idx][sim_close]))
                        pgt_index[idx][pos_c] = torch.cat((pgt_index[idx][pos_c], sim_close)).unique()

                        # 公式12 计算w，大括号里面就是sim hardness
                        sim_hardness = final_score_list[idx][sim_close, pos_c+1] / final_score_list[idx][:,pos_c+1].sum()
                        #sim_hardness = avg_score_split[idx][sim_close, pos_c+1] / avg_score_split[idx][:, pos_c+1].sum()
                        # 论文4.4 instance difficulty
                        instance_diff = torch.cat((instance_diff, sim_hardness.view(-1)))

            # 相似度loss * λ
            return_loss_dict['loss_sim'] = self.sim_lmda * self.sim_loss(pgt_update, instance_diff, device)

        # 计算其余各项损失
        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            # 该图片的图片级别标注？还是实例级别？
            labels_per_im = targets_per_im.get_field('labels').unique()
            # 包含C类的初始标签，全为0，这个C是图片含有的还是所有的？
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # image score 按行（即类）求和 将值限制在(0,1)之间
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            # WSDDN MIDN损失 交叉熵 每张图的图像得分和图像级别标注
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            # 多阶段OICR loss
            for i in range(num_refs):
                # 第1阶段的监督score是从WSDDN来的，其余阶段是从上一阶段来的
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                # 消融OD的时候进这里了，2.19疑问：进这里是不是意味着IoUSampling无论怎么设置都不会起效了。感觉再设置很麻烦不管了。
                if not self.contra and self.refine_p == 0:           ### oicr_layer ###
                    pseudo_labels, loss_weights, regression_targets = self.oicr_layer(
                        proposals_per_image, source_score, labels_per_im, device, return_targets=True
                        )
                elif not self.contra and self.refine_p > 0:          ### mist layer ###
                    pseudo_labels, loss_weights, regression_targets = self.mist_layer(
                        proposals_per_image, source_score, labels_per_im, device, return_targets=True
                        )
                # 进这个，生成伪标签
                elif self.contra and self.refine_p == 0:                ### od layer ###
                    pseudo_labels, loss_weights, regression_targets = self.od_layer(
                    proposals_per_image, source_score, labels_per_im, device, pgt_instance[idx][i], return_targets=True
                    )
                # False 这是针对point或scribble标签的
                if self.roi_refine:
                    pseudo_labels = self.filter_pseudo_labels(pseudo_labels, proposals_per_image, targets_per_im)

                lmda = 3 if i == 0 else 1

                # 分类损失
                return_loss_dict['loss_ref_cls%d'%i] += lmda * torch.mean(
                    F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights
                )

                # regression
                sampled_pos_inds_subset = torch.nonzero(pseudo_labels>0, as_tuple=False).squeeze(1)
                labels_pos = pseudo_labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

                box_regression = ref_bbox_preds[i][idx]
                reg_loss = lmda * torch.sum(smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    beta=1, reduction=False) * loss_weights[sampled_pos_inds_subset, None]
                )
                reg_loss /= pseudo_labels.numel()
                # 回归损失
                return_loss_dict['loss_ref_reg%d'%i] += reg_loss

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0   # 按图拆分的WSDDN proposal得分不能为空集，即bs！=0吧

        # 对各项损失值除以batch size，但sim loss不除
        for l in return_loss_dict.keys():
            # 为什么sim loss不用除以长度？
            if 'sim' in l:
                continue
            return_loss_dict[l] /= len(final_score_list)

        for a in return_acc_dict.keys():
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict


def make_roi_weak_loss_evaluator(cfg):
    # RoIRegLoss
    func = registry.ROI_WEAK_LOSS[cfg.MODEL.ROI_WEAK_HEAD.LOSS]
    return func(cfg)
