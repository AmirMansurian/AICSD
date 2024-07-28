import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from cbam import *

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, is_student = True):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.is_student = is_student

        if self.is_student:
            self.cbam_modules = None
            self.attn_modules = None
            self.ema_modules = None

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_bn_before_relu(self):
        BNs = self.backbone.get_bn_before_relu()
        BNs += self.aspp.get_bn_before_relu()
        BNs += self.decoder.get_bn_before_relu()

        return BNs

    def get_channel_num(self):
        channels = self.backbone.get_channel_num()
        channels += self.aspp.get_channel_num()
        channels += self.decoder.get_channel_num()

        return channels

    def extract_feature(self, input):
        feats, x, low_level_feat = self.backbone.extract_feature(input)
        feat, x = self.aspp.extract_feature(x)
        feats += feat
        feat, x = self.decoder.extract_feature(x, low_level_feat)
        feats += feat
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return feats, x
    

    def set_cbam_modules(self, cbam_modules):
        self.cbam_modules = cbam_modules

    def set_attn_modules(self, attn_modules):
        self.attn_modules = attn_modules

    def set_ema_modules(self, ema_modules):
        self.ema_modules = ema_modules
    
    def extract_cbam_features(self, input):
        feats, _ = self.extract_feature(input)
        feat_num = len(feats)

        if self.is_student:
            if self.cbam_modules is None:
                return None
            for i in range(3, feat_num):
                b,c,h,w = feats[i].shape
                feats[i] = self.cbam_modules[i-3](feats[i]).view(b, c, -1).detach()
                feats[i] = torch.nn.functional.normalize(feats[i], dim = 1)
        else:
            for i in range(3, feat_num):
                b,c,h,w = feats[i].shape
                feats[i] = CBAM(feats[i].shape[1], model = 'teacher').cuda()(feats[i]).view(b, c, -1).detach()
                feats[i] = torch.nn.functional.normalize(feats[i], dim = 1)

        return feats

