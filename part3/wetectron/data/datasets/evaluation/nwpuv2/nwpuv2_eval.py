# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from wetectron.structures.bounding_box import BoxList
from wetectron.structures.boxlist_ops import boxlist_iou


def do_nwpuv2_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    # 每张图和对应的预测结果，pred_boxlists记录的是某张图的预测结果，gt_boxlists记录的是这张图的gt
    # 所以CorLoc只要在上面两个的每一项里有起码一个就算一个
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
    # result 直接包含了最终输出的各项结果和map
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )

    result_rec="mrec: {:.4f}\n".format(result["mrec"])
    for i, rec in enumerate(result["rec"]):
        result_rec += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i+1), rec
        )
    logger.info(result_str)
    logger.info(result_rec)

    print(result_str)
    print(result_rec)

    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
            fid.write(result_rec)
    return result

'''
def do_voc_evaluation(dataset, predictions, output_folder, logger):
    class_boxes = {dataset.map_class_id_to_class_name(i + 1): [] for i in range(20)}
    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()

        for i, class_id in enumerate(pred_label):
            image_name = dataset.get_origin_id(image_id)
            box = pred_bbox[i]
            score = pred_score[i]
            class_name = dataset.map_class_id_to_class_name(class_id)
            class_boxes[class_name].append((image_name, box[0], box[1], box[2], box[3], score))
    aps = []
    tmp = os.path.join(output_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    for key in dataset.CLASSES[1:]:
        filename = os.path.join(output_folder, '{}.txt'.format(key))
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wt') as txt:
            boxes = class_boxes[key]
            for k in range(len(boxes)):
                box = boxes[k]
                txt.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(box[0], box[-1], box[1], box[2], box[3], box[4]))
        devkit_path = '/data7/lufficc/voc/VOCdevkit/VOC2007'
        annopath = os.path.join(devkit_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(devkit_path, 'ImageSets', 'Main', 'test.txt')
        rec, prec, ap = voc_eval(filename, annopath, imagesetfile, key, tmp, ovthresh=0.5, use_07_metric=True)
        aps += [ap]
        print(('AP for {} = {:.4f}'.format(key, ap)))
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

def get_origin_id(self, index):
    img_id = self.ids[index]
    return img_id
'''
def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."   # 说明每张图的预测结果和gt不对应，有的图没有其中一项
    # 应该在这里返回一个corloc
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    # 计算average precision
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    meanrec=[]
    for i in range(1, len(rec)):
        meanrec.append(np.mean(rec[i]))
    # return {"ap": ap, "map": np.nanmean(ap)}
    return {"ap": ap, "map": np.nanmean(ap), "rec": meanrec, "mrec": sum(meanrec)/len(meanrec)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    def defaultvalue():
        return [0,0]
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    corloc_pred = defaultdict(list)
    this_image_has_classl= defaultdict(list)
    this_image_match_classl= defaultdict(list)
    ratio= defaultdict(defaultvalue)

    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        if gt_boxlist.bbox.device.type == 'cuda' and pred_boxlist.bbox.device.type == 'cuda':
            pred_bbox = pred_boxlist.bbox.detach().cpu().numpy()
            gt_bbox = gt_boxlist.bbox.detach().cpu().numpy()
            pred_label = pred_boxlist.get_field("labels").detach().cpu().numpy()
            pred_score = pred_boxlist.get_field("scores").detach().cpu().numpy()
            gt_label = gt_boxlist.get_field("labels").detach().cpu().numpy()
            # gt_difficult = gt_boxlist.get_field("difficult").detach().cpu().numpy()
            gt_difficult = np.array([0] * len(gt_boxlist))
        else:
            pred_bbox = pred_boxlist.bbox.numpy()
            gt_bbox = gt_boxlist.bbox.numpy()
            pred_label = pred_boxlist.get_field("labels").numpy()
            pred_score = pred_boxlist.get_field("scores").numpy()
            gt_label = gt_boxlist.get_field("labels").numpy()
            # gt_difficult = gt_boxlist.get_field("difficult").numpy()
            gt_difficult=np.array([0]*len(gt_boxlist))
        #pred_score = pred_boxlist.get_field("scores").numpy()
        #pred_bbox = pred_boxlist.bbox.numpy()
        #pred_label = pred_boxlist.get_field("labels").numpy()
        #pred_score = pred_boxlist.get_field("scores").numpy()
        #gt_bbox = gt_boxlist.bbox.numpy()
        #gt_label = gt_boxlist.get_field("labels").numpy()
        #gt_difficult = gt_boxlist.get_field("difficult").numpy()

        # 遍历对于某张图，预测结果和gt中所有出现的类
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            # difficult为0的才拿来算
            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(gt_bbox_l):
                ratio[l][1] += 1

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue


            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            this_image_classl_flag=False
            for gt_idx in gt_index:
                if gt_idx >= 0:     # 成功匹配到gt的
                    if gt_difficult_l[gt_idx]:  # difficult的不算
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:   # 这个还没被分配过就认为和gt匹配
                            match[l].append(1)
                            if not this_image_classl_flag:
                                ratio[l][0]+=1
                                this_image_classl_flag=True
                        else:                   # 如果已经被分配过就是0
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)          # 未成功匹配gt，0


    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():  # keys就是所有的类
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:    # 不是difficult的
            rec[l] = tp / n_pos[l]
    # print(ratio)
    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
