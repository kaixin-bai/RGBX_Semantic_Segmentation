import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

# 设置分布式训练的MASTER_PORT环境变量
os.environ['MASTER_PORT'] = '169710'

# 启动计算引擎
with Engine(custom_parser=parser) as engine:
    # 解析命令行参数
    args = parser.parse_args()

    # 配置CUDA相关设置
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 数据加载器初始化，获取训练数据加载器和样本选择器
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # 设置模型和损失函数:设置交叉熵损失函数,根据是否进行分布式训练选择适当的BatchNorm层,初始化模型
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    # 分组权重和设置优化器:根据BatchNorm层的权重，对模型权重进行分组,根据配置文件中的参数设置优化器
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # 配置学习率策略
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    # 分布式训练的配置:根据是否进行分布式训练选择相应的设备，并调整模型
    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # 注册模型、优化器和数据加载器到计算引擎：这行代码向计算引擎注册了数据加载器、模型和优化器。这是为了在训练过程中保持这些组件的状态，便于后续的恢复和管理
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)

    # 恢复之前的训练状态：如果有可继续的训练状态（例如中断后的恢复），则从检查点恢复模型、优化器等的状态。
    if engine.continue_state_object:
        engine.restore_checkpoint()

    # 初始化优化器和模型的状态：清空（重置）优化器的梯度，将模型设置为训练模式（启用BatchNorm和Dropout等）
    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    # 开始训练的主循环:
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        # 分布式训练的采样器设置：如果在进行分布式训练，更新训练采样器的epoch值，确保每个epoch的数据划分一致
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        # 设置进度条：初始化进度条，并设置格式
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        # 初始化数据加载器迭代器：将数据加载器转换为迭代器，以便在循环中使用
        dataloader = iter(train_loader)
        # 初始化累计损失
        sum_loss = 0

        for idx in pbar:
            # 更新计算引擎的迭代信息：更新计算引擎中当前的epoch和迭代次数信息
            engine.update_iteration(epoch, idx)
            # 数据加载与预处理：从数据加载器获取一个minibatch，提取并处理需要的数据，如图像、标签等，将数据移动到GPU上进行加速计算
            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
            # 计算损失：设置辅助损失的比例，通过模型计算损失
            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)
            # 多GPU上损失的同步：如果是分布式训练环境，同步所有GPU上的损失
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            # 优化器梯度清零，反向传播和优化步骤：重置优化器的梯度，执行反向传播来计算梯度，更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新学习率：根据当前迭代次数计算学习率，更新优化器中的学习率
            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            # 打印训练信息：累计损失，根据是否是分布式训练，打印相应的训练信息
            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            # 释放loss的内存
            del loss
            # 更新进度条：设置进度条的描述信息
            pbar.set_description(print_str, refresh=False)
        # 记录TensorBoard：在分布式训练的主节点或非分布式环境下，将平均损失记录到TensorBoard
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
        # 保存checkpoint：根据配置的条件（比如达到某个epoch或达到保存频率），保存模型的检查点
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
