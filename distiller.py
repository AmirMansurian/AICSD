import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np

from att_modules.attn_types import attn_types

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


        # We ignore the intermediate feature maps from the encoder (first 3 layers).
        self.start_layer = 3
        self.end_layer = len(t_channels)

        ema_factor = 32

        # Mapping modules transform student feature maps into the teacher's feature space.
        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        self.attns = [attn_types[args.att_type](s_channels[i], model = 'student').cuda() 
                      for i in range(self.start_layer, len(s_channels))]
        self.attns = nn.ModuleList(self.attns)
                    

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args


    def forward(self, x, y):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)


        attn_loss = 0

        if self.args.att_lambda is not None:
            for i in range(self.start_layer, self.end_layer):
                b,c,h,w = t_feats[i].shape
                M = h * w

                s_feats_att = self.Connectors[i](self.attns[i - self.start_layer](s_feats[i])).view(b, c, -1)

                # Attention loss for different modules; teacher's modules are fixed and not trained.
                # Each module differs slightly between the teacher and student models to account for unlearnable parameters.
                t_feats_att = attn_types[self.args.att_type](t_feats[i].shape[1], model = 'teacher').cuda()(t_feats[i]).view(b, c, -1).detach()

                s_feats_att = torch.nn.functional.normalize(s_feats_att, dim=1)
                t_feats_att = torch.nn.functional.normalize(t_feats_att, dim=1)

                attn_loss += torch.norm(s_feats_att - t_feats_att, dim = 1).sum() / M * self.args.att_lambda
        
        # Naive KD loss
        kd_loss = 0

        if self.args.kd_lambda is not None:
          kd_loss = self.args.kd_lambda * torch.nn.KLDivLoss()(F.log_softmax(s_out / self.temperature, dim=1), 
                                                                F.softmax(t_out / self.temperature, dim=1))
          
        # Lad loss in https://ieeexplore.ieee.org/document/10484265/
        lad_loss = 0

        if self.args.lad_lambda is not None:
            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                lad_loss += (s_feats[i] / torch.norm(s_feats[i], p = 2) - t_feats[i] / torch.norm(t_feats[i], p = 2)).pow(2).sum() \
                    / (b) * self.args.lad_lambda

        # Pad loss in https://ieeexplore.ieee.org/document/10484265/
        pad_loss = 0

        if self.args.pad_lambda is not None:
            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                pad_loss += (F.normalize(s_feats[i], p = 2, dim = 1) - F.normalize(t_feats[i], p = 2, dim = 1)).pow(2).sum() \
                    / (h * w * b) * self.args.pad_lambda

        # Cad loss in https://ieeexplore.ieee.org/document/10484265/
        cad_loss = 0

        if self.args.cad_lambda is not None:
            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                cad_loss += (F.normalize(s_feats[i], p = 2, dim = (2,3)) - F.normalize(t_feats[i], p = 2, dim = (2,3))).pow(2).sum() \
                    / (c * b) * self.args.cad_lambda

        # Naive loss in https://ieeexplore.ieee.org/document/10484265/
        naive_loss = 0

        if self.args.naive_lambda is not None:

            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                s_feats[i] = self.Connectors[i](s_feats[i])
                
                naive_loss += (s_feats[i] - t_feats[i]).pow(2).sum() / (h * w * c* b) * self.args.naive_lambda


        return s_out, kd_loss , lad_loss , pad_loss , cad_loss , naive_loss, attn_loss

    
    # Used for feature attention maps after training
    def get_attn_modules(self):
        return self.attns