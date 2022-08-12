import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def dist_loss(source, target):
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

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        self.temperature = 1

    def forward(self, x):

        #print('Teacherrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        t_feats, t_out, dist_t = self.t_net.extract_feature(x)
        #print('Studentttttttttttttttttttttttttttttttttttttttttt')
        s_feats, s_out, dist_s = self.s_net.extract_feature(x)
        feat_num = len(t_feats)
        
        loss_distill = 0
        #print('########################################')
        for i in range(len(t_feats)):
          TF = F.normalize(t_feats[i].pow(2).mean(1)) 
          SF = F.normalize(s_feats[i].pow(2).mean(1)) 
          temp = (TF - SF).pow(2).mean()
          loss_distill += temp
        
       # gt = F.interpolate(t_feats[4], size=x.size()[2:], mode='bilinear', align_corners=True).pow(2).mean(1)
       # gs = F.interpolate(s_feats[4], size=x.size()[2:], mode='bilinear', align_corners=True).pow(2).mean(1)

       # grad_distill = 0
       # for i in range(21) :

         # grad_t = F.normalize(t_out[:, i] - gt)
         # grad_s = F.normalize(s_out[:, i] - gs)
          #print(grad_t.shape)
          #grad_distill += (grad_t - grad_s).pow(2).mean()

        #TF = F.normalize(t_feats[4].pow(2).mean(1)) 
        #SF = F.normalize(s_feats[4].pow(2).mean(1)) 
        #loss_distill = 0
        #loss_distill = (TF - SF).pow(2).mean()
        
        #loss_distill2 =  torch.nn.KLDivLoss()(F.log_softmax(s_out / self.temperature, dim=1), F.softmax(t_out / self.temperature, dim=1))


        return s_out, loss_distill
