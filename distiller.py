import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
from cbam import *
from da_att import *
import scipy

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

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        # self.cbams = nn.ModuleList([CBAM(s_channels[i], model = 'student').cuda() for i in range(len(s_channels))])
        # self.attn = PAM_Module(s_channels[3], 'student').cuda()

        self.attns = nn.ModuleList([CBAM(s_channels[i], model = 'student').cuda() for i in range(3, len(s_channels))])

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

        loss_distill = 0


        def overhaul():
            loss_distill = 0
            for i in range(feat_num):
                s_feats[i] = self.Connectors[i](s_feats[i])
                loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                                / self.loss_divider[i]
            return loss_distill
        

        loss_cbam = 0

        for i in range(3, feat_num):
            b,c,h,w = t_feats[i].shape
            M = h * w
            s_feats[i] = self.Connectors[i](self.attns[i-3](s_feats[i])).view(b, c, -1)
            t_feats[i] = CBAM(t_feats[i].shape[1], model = 'teacher').cuda()(t_feats[i]).view(b, c, -1).detach()

            s_feats[i] = torch.nn.functional.normalize(s_feats[i], dim = 1)
            t_feats[i] = torch.nn.functional.normalize(t_feats[i], dim = 1)

            loss_cbam += torch.norm(s_feats[i] - t_feats[i], dim = 1).sum() / M * 0.1


        return s_out, loss_cbam
    

    def get_cbam_modules(self):
        return self.attns