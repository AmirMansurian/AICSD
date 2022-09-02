import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category:, :, : ] * self.mask).sum()




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



class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


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
        self.t_net.eval()

        #print(t_net)

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        self.criterion = sim_dis_compute
        self.temperature = 1
        self.scale = 0.5


    def gcam(self, image, out, net, sem_class_to_idx, cls): 
      car_category = sem_class_to_idx[cls]
      car_mask = out[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
      car_mask_float = np.float32(car_mask == car_category)

      target_layers = [net.aspp.conv1]
      targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
      with GradCAM(model=net,
                        target_layers=target_layers,
                        use_cuda=torch.cuda.is_available()) as cam:
                grayscale_cam = cam(input_tensor=torch.unsqueeze(image, 0),
                                    targets=targets)[0, :]

      return grayscale_cam


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
   

        pi_loss = 0
        if self.args.pi_lambda is not None: # pixelwise loss
          TF = F.normalize(t_feats[5].pow(2).mean(1)) 
          SF = F.normalize(s_feats[5].pow(2).mean(1)) 
          pi_loss = self.args.pi_lambda * (TF - SF).pow(2).mean()


        lo_loss = 0
        if self.args.lo_lambda is not None: #logits loss
          lo_loss =  self.args.lo_lambda * torch.nn.KLDivLoss()(F.log_softmax(s_out / self.temperature, dim=1), F.softmax(t_out / self.temperature, dim=1))


        sem_classes = [
              '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
              'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
          ]
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        
        gcam_loss = 0
        if self.args.gcam_lambda is not None:
          for i in range(4) : 
            for cls in sem_classes : 
              t_cam = self.gcam(x[i], t_out, self.t_net, sem_class_to_idx, cls)
              s_cam = self.gcam(x[i], s_out, self.s_net, sem_class_to_idx, cls)
              gcam_loss += (F.normalize(torch.from_numpy(t_cam)) - F.normalize(torch.from_numpy(s_cam))).pow(2).mean()
          gcam_loss *= self.args.gcam_lambda  
         
            
        return s_out, pa_loss, pi_loss, lo_loss, gcam_loss
