# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm
import pickle
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from ..utils.visualize import vis_results
from .bbox_aug import im_detect_bbox_aug
import numpy as np


def compute_on_dataset(model, data_loader, device, timer=None,meta_attentions=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:   # False
                output = im_detect_bbox_aug(model, images, device, meta_attentions, targets)
            else:
                output = model(images.to(device), [target.to(device) for target in targets],meta_attentions=meta_attentions)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

            # 可视化耗时极小
            if cfg.TEST.VISUALIZE:
                data_path = data_loader.dataset.root
                img_infos = [data_loader.dataset.get_img_info(ind) for ind in image_ids]
                vis_results(output, img_infos, data_path, show_mask_heatmaps=False)

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )


    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.engine.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    # now = time.strftime("%Y-%m-%d-%H-%M-%S")
    # predictiontxt = open(str(now) + 'predictions.txt', 'w')
    #
    # for image_id, prediction in enumerate(predictions):
    #     predictiontxt.write(str(image_id)+' : '+str(prediction) + '\n')
    # predictiontxt.close()
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        phase=1,
        shot=10,
        split=1,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    print('loading mean class attentions')
    input_dir = "saved_attentions/"
    torch.cuda.empty_cache()
    meta_attentions = pickle.load(open(os.path.join(
        input_dir, 'meta_type_{}'.format(split),
        str(phase) + '_shots_' + str(shot) + '_mean_class_attentions.pkl'), 'rb'))
 
    predictions = compute_on_dataset(model, data_loader, device, inference_timer,meta_attentions)

    # wait for all processes to complete before measuring the time
    synchronize()
    
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    # zyc 22.4.30 add
    if predictions:
        with open(os.path.join(input_dir, 'prediction_results_'+str(split)+'_'+str(shot)+'.txt'), 'w') as f:
            for ind, pre in enumerate(predictions):
                thisbbox=pre.bbox
                if len(thisbbox):
                    for j in thisbbox:
                        if len(j)==4:
                            xmin,ymin,xmax,ymax=j.numpy()
                            s=str(ind)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+'\n'
                        else:
                            xmin, ymin, xmax, ymax=0,0,0,0
                            s=str(ind)+' 0\n'
                        f.write(s)
                else:
                    f.write(str(ind)+' 0\n')


    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))


    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
