'''
# author: Zhiyuan Yan (修改版)
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706 (修改: 2025-05-09)
# description: 适用于张量输入的UCFDetector类

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. features: Feature-extraction
4. classifier: Classification
5. forward: Forward-propagation - 已优化以适用于张量直接输入

Reference:
@article{yan2023ucf,
  title={UCF: Uncovering Common Features for Generalizable Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2304.13949},
  year={2023}
}
'''

import os
import datetime
import logging
import random
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from utils.detectors.base_detector import AbstractDetector
from utils.detectors import DETECTOR
from utils.networks import BACKBONE

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='ucf')
class UCFDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['backbone_config']['num_classes']
        self.encoder_feat_dim = config['encoder_feat_dim']
        self.half_fingerprint_dim = self.encoder_feat_dim // 2

        self.encoder_f = self.build_backbone(config)
        self.encoder_c = self.build_backbone(config)

        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # conditional gan
        self.con_gan = Conditional_UNet()

        # head
        specific_task_number = len(config['train_dataset']) + 1  # default: 5 in FF++
        self.head_spe = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=specific_task_number
        )
        self.head_sha = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=self.num_classes
        )
        self.block_spe = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )
        self.block_sha = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone

    def features(self, input_tensor):
        """
        提取特征向量

        Args:
            input_tensor: 输入张量 [B, C, H, W]

        Returns:
            包含伪造特征和内容特征的字典
        """
        # encoder
        f_all = self.encoder_f.features(input_tensor)
        c_all = self.encoder_c.features(input_tensor)
        feat_dict = {'forgery': f_all, 'content': c_all}
        return feat_dict

    def classifier(self, features):
        """
        特征分类

        Args:
            features: 输入特征

        Returns:
            特定特征和共享特征
        """
        # 将特征分为特定特征和共享特征
        f_spe = self.block_spe(features)
        f_share = self.block_sha(features)
        return f_spe, f_share

    def forward(self, x):
        """
        前向传播函数 - 已优化为直接接受张量输入

        Args:
            x: 输入张量 [B, C, H, W]

        Returns:
            模型预测概率
        """
        # 提取特征
        features = self.features(x)
        forgery_features, content_features = features['forgery'], features['content']

        # 获取特定特征和共享特征
        f_spe, f_share = self.classifier(forgery_features)

        # 预测
        out_sha, sha_feat = self.head_sha(f_share)
        out_spe, spe_feat = self.head_spe(f_spe)

        # 计算虚假概率
        prob_sha = torch.softmax(out_spe, dim=1)[:, 1]

        return prob_sha


def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )


def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def c_norm(self, x, bs, ch, eps=1e-7):
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) \
              * y_std.expand(size) + y_mean.expand(size)
        return out


class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self):
        super(Conditional_UNet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)

        self.adain3 = AdaIN()
        self.adain2 = AdaIN()
        self.adain1 = AdaIN()

        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.activation = nn.Tanh()

    def forward(self, c, x):  # c is the style and x is the content
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)

        x = self.adain2(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up2(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up2(c)

        x = self.adain1(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up1(x)

        x = self.conv_last(x)
        out = self.up_last(x)

        return self.activation(out)


class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_f), )

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, out_f, 1, 1), )

    def forward(self, x):
        x = self.conv2d(x)
        return x


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_f), )

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat