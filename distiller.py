import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

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

def compute_fsp(g , f_size):
        fsp_list = []
        for i in range(f_size):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list

def compute_fsp_loss(s, t):
        return (s - t).pow(2).mean()

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
        self.criterion = sim_dis_compute
        self.temperature = 1
        self.scale = 0.5

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        pa_loss = 0 
        if self.args.pa_lambda is not None: # pairwise loss
          feat_T = t_feats[4]
          feat_S = s_feats[4]
          total_w, total_h = feat_T.shape[2], feat_T.shape[3]
          patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
          maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
          pa_loss = self.args.pa_lambda * self.criterion(maxpool(feat_S), maxpool(feat_T))

        sp_loss = 0 
        if self.args.sp_lambda is not None: # pairwise loss
          feat_T = t_feats[4]
          feat_S = s_feats[4]

          bsz = feat_S.shape[0]
          feat_S = feat_S.view(bsz, -1)
          feat_T = feat_T.view(bsz, -1)
          
          G_s = torch.mm(feat_S, torch.t(feat_S))        
          G_s = torch.nn.functional.normalize(G_s)
          G_t = torch.mm(feat_T, torch.t(feat_T))
          G_t = torch.nn.functional.normalize(G_t)

          G_diff = G_t - G_s
          sp_loss = self.args.sp_lambda * (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)    

        fsp_loss = 0 
        if self.args.fsp_lambda is not None: # pairwise loss
          bot_t, top_t = t_feats[4], t_feats[5]
          bot_s, top_s = s_feats[4], s_feats[5]

          b_H_t, t_H_t = bot_t.shape[2], top_t.shape[2]
          b_H_s, t_H_s = bot_s.shape[2], top_s.shape[2]

          if b_H_t > t_H_t:
              bot_t = F.adaptive_avg_pool2d(bot_t, (t_H_t, t_H_t))
          elif b_H_t < t_H_t:
              top_t = F.adaptive_avg_pool2d(top_t, (b_H_t, b_H_t))
          
          if b_H_s > t_H_s:
              bot_s = F.adaptive_avg_pool2d(bot_s, (t_H_s, t_H_s))
          elif b_H_s < t_H_s:
              top_s = F.adaptive_avg_pool2d(top_s, (b_H_s, b_H_s))

          bot_t = bot_t.unsqueeze(1)
          top_t = top_t.unsqueeze(2)
          bot_s = bot_s.unsqueeze(1)
          top_s = top_s.unsqueeze(2)

          bot_t = bot_t.view(bot_t.shape[0], bot_t.shape[1], bot_t.shape[2], -1)
          top_t = top_t.view(top_t.shape[0], top_t.shape[1], top_t.shape[2], -1)
          bot_s = bot_s.view(bot_s.shape[0], bot_s.shape[1], bot_s.shape[2], -1)
          top_s = top_s.view(top_s.shape[0], top_s.shape[1], top_s.shape[2], -1)

          fsp_t = (bot_t * top_t).mean(-1)
          fsp_s = (bot_s * top_s).mean(-1)


          fsp_t = torch.nn.functional.normalize(fsp_t)
          fsp_s = torch.nn.functional.normalize(fsp_s)

          fsp_loss =  self.args.fsp_lambda * (fsp_s - fsp_t).pow(2).mean()
          # fsp_loss = self.args.fsp_lambda * [compute_fsp_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
          
   

        pi_loss = 0
        if self.args.pi_lambda is not None: # pixelwise loss
          TF = F.normalize(t_feats[5].pow(2).mean(1)) 
          SF = F.normalize(s_feats[5].pow(2).mean(1)) 
          pi_loss = self.args.pi_lambda * (TF - SF).pow(2).mean()
        
        
        ic_loss = 0
        if self.args.ic_lambda is not None: #logits loss
          b, c, h, w = s_out.shape
          s_logit = torch.reshape(s_out, (b, c, h*w))
          t_logit = torch.reshape(t_out, (b, c, h*w))

          ICCT = torch.bmm(t_logit, t_logit.permute(0,2,1))
          ICCT = torch.nn.functional.normalize(ICCT, dim = 2)

          ICCS = torch.bmm(s_logit, s_logit.permute(0,2,1))
          ICCS = torch.nn.functional.normalize(ICCS, dim = 2)

          G_diff = ICCS - ICCT
          lo_loss = self.args.ic_lambda * (G_diff * G_diff).view(b, -1).sum() / (c*b)
        
        
        
        lo_loss = 0
        if self.args.lo_lambda is not None: #logits loss
          #lo_loss =  self.args.lo_lambda * torch.nn.KLDivLoss()(F.log_softmax(s_out / self.temperature, dim=1), F.softmax(t_out / self.temperature, dim=1))
          b, c, h, w = s_out.shape
          s_logit_t = torch.reshape(s_out, (b, c, h*w))
          t_logit_t = torch.reshape(t_out, (b, c, h*w))

          s_logit = F.softmax(s_logit_t / self.temperature, dim=2)
          t_logit = F.softmax(t_logit_t / self.temperature, dim=2)
          kl = torch.nn.KLDivLoss(reduction="batchmean")
          ICCS = torch.empty((21,21)).cuda()
          ICCT = torch.empty((21,21)).cuda()
          for i in range(21):
            for j in range(i, 21):
              ICCS[j, i] = ICCS[i, j] = kl(s_logit[:, i], s_logit[:, j])
              ICCT[j, i] = ICCT[i, j] = kl(t_logit[:, i], t_logit[:, j])

          ICCS = torch.nn.functional.normalize(ICCS, dim = 1)
          ICCT = torch.nn.functional.normalize(ICCT, dim = 1)
          lo_loss =  self.args.lo_lambda * (ICCS - ICCT).pow(2).mean()/b 
        
        kd_loss = pa_loss + pi_loss + ic_loss + lo_loss + sp_loss + fsp_loss
        return s_out, kd_loss
