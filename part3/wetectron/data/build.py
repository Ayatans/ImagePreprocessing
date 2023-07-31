# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import numpy as np

import torch.utils.data
from wetectron.utils.comm import get_world_size
from wetectron.utils.imports import import_file
from wetectron.utils.miscellaneous import save_labels, seed_all_rng
from wetectron.utils.model_zoo import cache_url

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True, proposal_files=None, min_size=None):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    # 进这个if，但len不是0
    if proposal_files is not None:
        if len(proposal_files) == 0:
            proposal_files = (None, ) * len(dataset_list)
    #assert len(dataset_list) == len(proposal_files)

    datasets = []
    data_args = []

    for index, dataset_name in enumerate(dataset_list):
        is_labeled = "unlabeled" not in dataset_name
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = (is_train and is_labeled)
        if data["factory"] == "PascalVOCDataset" or data["factory"] == 'DIORDataset' or data["factory"]== 'NWPUDataset' or data["factory"]== 'NWPUv2Dataset':
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        args["min_size"] = min_size

        # load proposal
        if proposal_files is not None:
            if 'voc' in dataset_name:
                _f = proposal_files[index]
            elif 'dior' in dataset_name:
                _f = proposal_files[index]
            elif 'coco' in dataset_name:
                _f = proposal_files[index]
            elif 'nwpuv2' in dataset_name:
                _f = proposal_files[index]
            elif 'nwpu' in dataset_name:
                _f = proposal_files[index]
            elif 'flickr' in dataset_name:
                _f = proposal_files[index]

            if _f is not None and _f.startswith("http"):
                # if the file is a url path, download it and cache it
                _f = cache_url(_f)
            args["proposal_file"] = _f

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)
        data_args.append(args)
    # for testing, return a list of datasets
    if not is_train:
        return datasets, data_args

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset], [data_args]


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
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        #target = dataset.get_groundtruth(i) ###
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, batch_size, args, class_batch, num_iters=None, start_iter=0
):

    # True, 进if
    if aspect_grouping:
        # class_batch=False，进else。else其实和if里没有区别。
        if class_batch == True:
            if not isinstance(aspect_grouping, (list, tuple)):
                aspect_grouping = [aspect_grouping]     # [1] 作为高：宽的标兵值，
            aspect_ratios = _compute_aspect_ratios(dataset)
            group_ids = _quantize(aspect_ratios, aspect_grouping)   # 传入list， list
            batch_sampler = samplers.GroupedBatchSampler(
                sampler, group_ids, images_per_batch, batch_size, dataset, class_batch, data_args=args, drop_uneven=False
            )
        else :
            if not isinstance(aspect_grouping, (list, tuple)):
                aspect_grouping = [aspect_grouping]     # [1]
            aspect_ratios = _compute_aspect_ratios(dataset)
            group_ids = _quantize(aspect_ratios, aspect_grouping)
            batch_sampler = samplers.GroupedBatchSampler(
                sampler, group_ids, images_per_batch, batch_size, dataset, class_batch, data_args=args, drop_uneven=False
            )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )

    # 训练时就不是none，进
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )

    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    # seed = cfg.SEED
    # def _init_fn(worker_id):
    #     np.random.seed(int(seed))
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER     # 几万
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
        )
        '''
    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    # True, so [1]
    aspect_grouping = [1.3] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else [] # dior改成1.3

    paths_catalog = import_file(
        "wetectron.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    proposal_files = cfg.PROPOSAL_FILES.TRAIN if is_train else cfg.PROPOSAL_FILES.TEST
    min_size = cfg.min_size

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    datasets, data_args = build_dataset(dataset_list, transforms, DatasetCatalog, is_train, proposal_files, min_size)
    class_batch = cfg.SOLVER.CLASS_BATCH

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)
    if not is_train:
        class_batch = False
    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, images_per_batch, data_args,
            class_batch, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            worker_init_fn=worker_init_reset_seed,
        )
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)