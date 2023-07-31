from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# 我用的torch1.4.0 会一直报warning，用来取消warning。1.1.0则不会有。
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')



def run_test(cfg, model, distributed,phase, shot, split):
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
        # default="configs/noveltest/e2e_dior_split1_30shot_finetune.yaml",
        default="configs/noveltest/e2e_nwpu_split1_5shot_finetune.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    args.distributed = num_gpus > 1  # True
    # 设置多卡，不用看
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # 根据超参的命名确定phase
    if "base" in args.config_file:
        phase = 1
    else:
        phase = 2

    split = args.config_file.split('split')[1][0]

    cfg.merge_from_file(args.config_file)  # 没细看，应该是将超参的yaml融入cfg吧
    cfg.merge_from_list(args.opts)  # 应该是将超参的opts融入cfg吧
    cfg.freeze()  # 冻结cfg

    # cfg.MODEL.SEED = args.seed
    output_dir = cfg.OUTPUT_DIR  # '.'
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))  # line1
    logger.info(args)  # line2

    # logger.info("Collecting env info (might take some time)")   # line3
    # logger.info("\n" + collect_env_info())  # 一堆系统配置，不需要展示，可注释

    # 展示config file名与内容，不需要展示，可注释
    # logger.info("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as cf:
    #     config_str = "\n" + cf.read()
    #     logger.info(config_str)

    # logger.info("Running with config:\n{}".format(cfg))

    model = build_detection_model(cfg)  # model=GeneralizedRCNN(cfg)
    device = torch.device(cfg.MODEL.DEVICE)  # 'cuda'
    model.to(device)  # 传入显存

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"  # DTYPE=float32 so = False
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'  # so O0
    model = amp.initialize(model,  opt_level=amp_opt_level)

    # True 设置多卡
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR  # '.'

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)  # "catalog://ImageNetPretrained/MSRA/R-101"
    arguments.update(extra_checkpoint_data)  # 读取R101预训练权重

    shot = cfg.INPUT.SHOTS  # 400
    print('before test')
    run_test(cfg, model, args.distributed, phase, shot, split)


if __name__ == "__main__":
    main()
