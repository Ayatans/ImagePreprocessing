# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms
import sys

def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True, shots=200, size=224,seed=0):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
            args["seed"] = seed
        if data["factory"] == "DIORDataset":
            args["use_difficult"] = not is_train
            args["seed"] = seed
        if data["factory"]=='NWPUDataset':
            args["use_difficult"] = not is_train
            args["seed"] = seed
        args["transforms"] = transforms
        if data["factory"] == "PascalVOCDataset_Meta":
            args["transforms"] = None
            args["shots"] = shots
            args["size"] = size
            args["seed"] = seed
        if data["factory"] == "DIORDataset_Meta":
            args["transforms"] = None
            args["shots"] = shots
            args["size"] = size
            args["seed"] = seed
        if data["factory"] == "NWPUDataset_Meta":
            args["transforms"] = None
            args["shots"] = shots
            args["size"] = size
            args["seed"] = seed
        if data["factory"] == "COCODataset_Meta":
            args["transforms"] = None
            args["shots"] = shots
            args["size"] = size
            args["remove_images_without_annotations"] = is_train
             
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    # 返回长度=aspect ratio，其中值为1的地方说明原ratio>=1，值为0处说明原ratio<1
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping: # [1]
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset) # len=7596

        # 检测ratios时用的
        # with open('/data3/yczhang/code/DCNet/experiments/DRD/temp.txt', 'a') as f:
        #     f.write(str(aspect_ratios)+'\n')

        # 返回长度=aspect ratio，其中值为1的地方说明原ratio>=1，值为0处说明原ratio<1
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, meta=False):
    num_gpus = get_world_size()
    if is_train:    # True
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH     # yaml定义 4
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        # 11.26 注释了 嫌烦
        '''
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
            # Equivalent schedules with...
            # 1 GPU:
            #   BASE_LR: 0.0025
            #   MAX_ITER: 60000
            #   STEPS: [0, 30000, 40000]
            # 2 GPUs:
            #   BASE_LR: 0.005
            #   MAX_ITER: 30000
            #   STEPS: [0, 15000, 20000]
            # 4 GPUs:
            #   BASE_LR: 0.01
            #   MAX_ITER: 15000
            #   STEPS: [0, 7500, 10000]
            # 8 GPUs:
            #   BASE_LR: 0.02
            #   MAX_ITER: 7500
            #   STEPS: [0, 3750, 5000]
        )
        '''
    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1.3] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []   # [1]

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    dataset_list = cfg.DATASETS.META if meta else dataset_list
    if meta == True:    # meta的话，强制每个gpu1张图
        images_per_gpu = 1
    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)    # enabled始终false所以有trans
    shots = cfg.INPUT.SHOTS
    size = cfg.INPUT.META_SIZE
    seed = cfg.MODEL.SEED
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train, shots, size, seed)
    if 'standard' in dataset_list[0] or 'meta' in dataset_list[0]:
        aspect_grouping = False
    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY,meta=meta)
        num_workers = cfg.DATALOADER.NUM_WORKERS
     
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
