#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    """
    将一个神经网络模型中的参数分组，以便在优化器中对不同的参数组应用不同的学习率和权重衰减（weight decay）。这种方法通常用于实现对某些层（如批归一化层）使用不同的正则化策略
    这个函数在实践中非常有用，尤其是当使用诸如Adam之类的优化器时，对于不同类型的层使用不同的正则化策略可以提高模型性能
    """
    # 定义分组列表：group_decay 用于存储需要应用权重衰减的参数；group_no_decay 用于存储不需要权重衰减的参数
    group_decay = []
    group_no_decay = []
    # 初始化计数变量：初始化一个计数器，但在此函数中似乎未被使用
    count = 0
    # 遍历模型中的所有模块：遍历传入模型中的所有子模块
    for m in module.modules():
        # 处理线性层（全连接层）：如果模块是线性层（全连接层），则将其权重添加到需要衰减的组中；如果这个层有偏置，则将偏置添加到不需要衰减的组中
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        # 处理卷积层：如果模块是卷积层（包括传统卷积层和转置卷积层），类似地处理其权重和偏置
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        # 处理归一化层：如果模块是归一化层（包括自定义的归一化层和常见的归一化类型），将其参数放入不需要衰减的组
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        # 处理其他 nn.Parameter 类型参数：如果模块是 nn.Parameter 类型，将其添加到需要衰减的组中
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)
    # 断言：确保所有参数都被考虑
    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    # 添加需要衰减的参数组：将需要衰减的参数组添加到 weight_group 中，并指定学习率
    weight_group.append(dict(params=group_decay, lr=lr))
    # 添加不需要衰减的参数组：将不需要衰减的参数组添加到 weight_group 中，设置 weight_decay 为 0，指定学习率
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    # 返回更新后的参数组：返回更新后的 weight_group，包含两种类型的参数组
    return weight_group
