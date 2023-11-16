import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
from cbam import *
from dist_kd import DIST   
from da_att import *
from kd import CriterionKD
from cirkd_memory import StudentSegContrast
from cirkd_mini_batch import CriterionMiniBatchCrossImagePair

import math

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)



class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        # self.crit = nn.CrossEntropyLoss(size_average = True).cuda()
        # self.kd_crit = DIST().cuda()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])


        img_rand = torch.rand(2, 3, 512, 512)
        feat_t, _ = t_net.extract_feature(img_rand)
        num_class = feat_t[-1].shape[1]


        self.criterion_kd = CriterionKD(temperature=1.0).cuda()
        self.criterion_minibatch = CriterionMiniBatchCrossImagePair(temperature=0.1).cuda()
        self.criterion_memory_contrast = StudentSegContrast(num_classes=num_class,
                                                     pixel_memory_size=20000,
                                                     region_memory_size=2000,
                                                     region_contrast_size=1024//num_class+1,
                                                     pixel_contrast_size=4096//num_class+1,
                                                     contrast_kd_temperature=1.0,
                                                     contrast_temperature=0.1,
                                                     s_channels=s_channels[-2],
                                                     t_channels=t_channels[-2], 
                                                     ignore_label=255).cuda()

        self.attns = nn.ModuleList([CBAM(s_channels[i], model = 'student').cuda() for i in range(3, len(s_channels))])
        
        self.temperature = 1

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]

    def forward(self, x, y):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)

        feat_num = len(t_feats)

        loss_cbam = 0

        # for i in range(3, feat_num):
        #     b,c,h,w = t_feats[i].shape
        #     M = h * w
        #     s_feats[i] = self.Connectors[i](self.attns[i-3](s_feats[i])).view(b, c, -1)
        #     t_feats[i] = CBAM(t_feats[i].shape[1], model = 'teacher').cuda()(t_feats[i]).view(b, c, -1).detach()

        #     s_feats[i] = torch.nn.functional.normalize(s_feats[i], dim = 1)
        #     t_feats[i] = torch.nn.functional.normalize(t_feats[i], dim = 1)

        #     loss_cbam += torch.norm(s_feats[i] - t_feats[i], dim = 1).sum() / M * 0.1

        kd_loss = self.criterion_kd(s_out, t_out)
        minibatch_pixel_contrast_loss = self.criterion_minibatch(s_feats[-2], t_feats[-2])

        _, predict = torch.max(s_out, dim=1) 
        memory_pixel_contrast_loss, memory_region_contrast_loss = \
            self.criterion_memory_contrast(s_feats[-2], t_feats[-2].detach(), y, predict)
        
        return s_out, kd_loss, minibatch_pixel_contrast_loss, \
            memory_pixel_contrast_loss, memory_region_contrast_loss