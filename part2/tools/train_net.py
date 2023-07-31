# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
# 我用的torch1.4.0 会一直报warning，用来取消warning。1.1.0则不会有。
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import random 

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')



def train(cfg, local_rank, distributed, phase, shot, split):
    model = build_detection_model(cfg)  # model=GeneralizedRCNN(cfg)
    device = torch.device(cfg.MODEL.DEVICE) # 'cuda'
    model.to(device)    # 传入显存


    optimizer = make_optimizer(cfg, model)  # 构建优化器 sgd 0.005 decay0.0001
    scheduler = make_lr_scheduler(cfg, optimizer)   # 优化器到指定步数时衰减lr

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"    # DTYPE=float32 so = False
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'   # so O0
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    # True 设置多卡
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,find_unused_parameters=True,
        )

          
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR     # '.'

    save_to_disk = get_rank() == 0
    # 这里加载模型了
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    # 这里读取了预训练的权重或微调时的base权重
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT) # "catalog://ImageNetPretrained/MSRA/R-101"
    arguments.update(extra_checkpoint_data) # 读取R101预训练权重

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    # meta=True，则强制每个gpu给1张图，无视bs设置。所以meta是query
    meta_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        meta=True
    )
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD    # 2500 后面每2500个iter就保存一个pth
    do_train(
        model,
        data_loader,
        meta_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        phase,
        shot,
        split,
        arguments,
        cfg
    )

    return model


def run_test(cfg, model, distributed,phase, shot, split, use_best):
    if distributed: # True
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)


    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            phase = phase,
            shot = shot,
            split = split,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )   # sh中没设定，说明是false，说明train后接test
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    args.distributed = num_gpus > 1     # True
    # 设置多卡，不用看
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    # 根据超参的命名确定phase
    if("base" in args.config_file):
        phase = 1
    else:
        phase = 2

    split = args.config_file.split('split')[1][0]

    cfg.merge_from_file(args.config_file)   # 没细看，应该是将超参的yaml融入cfg吧
    cfg.merge_from_list(args.opts)  # 应该是将超参的opts融入cfg吧
    cfg.freeze()    # 冻结cfg
    args.seed=cfg.MODEL.SEED    # 3.21 23:46 add
    random.seed(args.seed)  # 3.20 15:34 add
    np.random.seed(args.seed)

    #cfg.MODEL.SEED = args.seed
    output_dir = cfg.OUTPUT_DIR     # '.'
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))   # line1
    logger.info(args)   # line2

    #logger.info("Collecting env info (might take some time)")   # line3
    #logger.info("\n" + collect_env_info())  # 一堆系统配置，不需要展示，可注释

    # 展示config file名与内容，不需要展示，可注释
    #logger.info("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as cf:
    #     config_str = "\n" + cf.read()
    #     logger.info(config_str)

    # logger.info("Running with config:\n{}".format(cfg))
    shot = cfg.INPUT.SHOTS  # 400
    print('before training')
    model = train(cfg, args.local_rank, args.distributed, phase, shot, split)

    torch.cuda.empty_cache()

    if not args.skip_test:  # train后接测试
        run_test(cfg, model, args.distributed, phase, shot, split, use_best=False)
        # bestmodel = build_detection_model(cfg)  # model=GeneralizedRCNN(cfg)
        # device = torch.device(cfg.MODEL.DEVICE)  # 'cuda'
        # bestmodel.to(device)  # 传入显存
        # checkpointer = DetectronCheckpointer(cfg, bestmodel, save_dir=output_dir)
        # _ = checkpointer.load(cfg.MODEL.WEIGHT)
        # run_test(cfg, bestmodel, args.distributed, phase, shot, split, use_best=True)


if __name__ == "__main__":
    main()
