import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math

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
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args
        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        self.temperature = 1
        self.scale = 0.5

    def forward(self, x):

        t_feats, t_attens, t_out = self.t_net.extract_feature(x)
        s_feats, s_attens, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)
        
        loss_cbam = 0
        for i in range(3, feat_num):
            b,c,h,w = t_feats[i].shape
            #s_feats[i] = self.Connectors[i](s_feats[i])
            #lad_loss += (s_feats[i] / torch.norm(s_feats[i], p = 2) - t_feats[i] / torch.norm(t_feats[i], p = 2)).pow(2).sum() / (b) * 10

            s_attens[i-3] = self.Connectors[i](s_attens[i-3])
            s_attens[i-3] = torch.nn.functional.normalize(s_attens[i-3], dim = 1)
            t_attens[i-3] = torch.nn.functional.normalize(t_attens[i-3], dim = 1)

            loss_cbam += torch.norm(s_attens[i-3] - t_attens[i-3], dim = 1).sum() / (h * w * b) * 0.1
        

        return s_out, loss_cbam
