# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import pickle
import os
import torch
import torch.distributed as dist
import collections
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单卡就不需处理
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
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
    cfg,
):
    logger = logging.getLogger("maskrcnn_benchmark.engine.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    #writer = SummaryWriter('/root/code/zyc/DCNet/experiments/DRD/summarylog/')

    
    start_iter = arguments["iteration"] # 从头训是0
    model.train()
    start_training_time = time.time()
    end = time.time()
    # best_loss=999
    # print(len(meta_loader))   # =cfg中的MAX_ITER
    # contra_loss_weight_decay=cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY
    # decay_steps=cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
    data_iter_meta = iter(meta_loader)
    print('meta itered')
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):  # 从start_iter下标处开始枚举
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        scheduler.step()
        images = images.to(device)
        try:
            # nwpu: len(next(data_iter_meta))=2 tuple, next(data_iter_meta)[1].shape=[4, 256, 256]
            # [0]=list len=4 [0][0]~[0][3]=[1, 256, 256]
            # dior: len(next(data_iter_meta))=1 list , next(data_iter_meta)[0].shape=[20, 4, 256, 256]
            meta_input = next(data_iter_meta)[0].to(device)
        except:
            meta_iter = iter(meta_loader)   # 如果前面将所有meta loader数据用完，就重新用的意思是吗？
            meta_input = next(meta_iter)[0].to(device)
       
        # num_classes = meta_input.shape[0]

        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets, meta_input, iteration=iteration)  # 返回包含各loss名为key，对应loss值为value的loss 字典

        # print('before',loss_dict)
        # if contra_loss_weight_decay:
        #     if iteration in decay_steps:
        #         loss_dict['loss_contrastive']=torch.div(loss_dict['loss_contrastive'],2.0)
        # print('after', loss_dict)

        losses = sum(loss for loss in loss_dict.values())   # 所有loss求和

        # torch.cuda.empty_cache()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict) # 将loss换到gpu0上？反正是一个在gpu转移loss的操作
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced) # loss是总的loss，后面跟所有细节loss
        #writer.add_scalar('loss-iter', losses, iteration)
        optimizer.zero_grad()

        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            # torch.cuda.empty_cache()
            scaled_losses.backward()
        optimizer.step()


        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "max_iter: {max_iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter=max_iter,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        # if losses_reduced<best_loss and iteration>2000:
        #     best_loss=losses_reduced
        #     print('best loss: ', best_loss)
        #     checkpointer.save("model_best", **arguments)
            # with torch.no_grad():
            #     class_attentions_tmp = collections.defaultdict(list)
            #     meta_iter1_tmp = iter(meta_loader)
            #     for i in range(shot):
            #         meta_input_tmp = next(meta_iter1_tmp)[0]
            #         num_classes_tmp = meta_input_tmp.shape[0]
            #
            #         meta_label_tmp = []
            #         for n in range(num_classes_tmp):
            #             meta_label_tmp.append(n)
            #         attentions_tmp = model(images, targets, meta_input, meta_label_tmp, average_shot=True)
            #         for idx in meta_label_tmp:
            #             class_attentions_tmp[idx].append(attentions_tmp[idx])
            # mean_class_attentions_tmp = {k: sum(v) / len(v) for k, v in class_attentions_tmp.items()}
            # output_dir = 'saved_attentions/'
            # save_path = os.path.join(output_dir, 'meta_type_{}'.format(split))
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # with open(os.path.join(save_path,
            #                        str(phase) + '_shots_' + str(shot) + '_best_mean_class_attentions.pkl'), 'wb') as f:
            #     pickle.dump(mean_class_attentions_tmp, f, pickle.HIGHEST_PROTOCOL)
            # print('save ' + str(shot) + ' best mean classes attentions done!')
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
    

    with torch.no_grad():
        class_attentions = collections.defaultdict(list)
        meta_iter1 = iter(meta_loader)
        for i in range(shot):
            try:
                meta_input = next(meta_iter1)[0].to(device)
            except:
                meta_iter1 = iter(meta_loader)  # 如果前面将所有meta loader数据用完，就重新用的意思是吗？
                meta_input = next(meta_iter1)[0].to(device)
            # meta_input = next(meta_iter1)[0]  # 不用上面的try的话会抽空，报错stopiteration
            num_classes = meta_input.shape[0]
          
            meta_label = []
            for n in range(num_classes):
                meta_label.append(n)
            attentions = model(images, targets, meta_input, meta_label,average_shot=True)
            for idx in meta_label:
                class_attentions[idx].append(attentions[idx])
    mean_class_attentions = {k: sum(v) / len(v) for k, v in class_attentions.items()}
    output_dir = 'saved_attentions/'
    save_path = os.path.join(output_dir, 'meta_type_{}'.format(split))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path,
                            str(phase) + '_shots_' + str(shot) + '_mean_class_attentions.pkl'), 'wb') as f:
        pickle.dump(mean_class_attentions, f, pickle.HIGHEST_PROTOCOL)
    print('save ' + str(shot) + ' mean classes attentions done!')                    
    time.sleep(5)

    torch.cuda.empty_cache()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


