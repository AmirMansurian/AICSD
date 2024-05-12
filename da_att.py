###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, Identity
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'Self_Att']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, model):
        super(PAM_Module, self).__init__()

        if model == 'student':
            self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
            self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
            self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.gamma = Parameter(torch.zeros(1)).cuda()
        else:
            self.query_conv = Identity()
            self.key_conv = Identity()
            self.value_conv = Identity()
            self.gamma = torch.ones(1).cuda()
        self.model = model

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, model):
        super(CAM_Module, self).__init__()

        if model == 'student':
            self.gamma = Parameter(torch.zeros(1)).cuda()
        else:
            # self.gamma = Parameter(torch.zeros(1))
            self.gamma = torch.ones(1).cuda()
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    

class Self_Att(Module):
    def __init__(self, in_dim, model = 'student'):
        super(Self_Att, self).__init__()
        self.PAM = PAM_Module(in_dim, model)
        self.CAM = CAM_Module(model)

        if model == 'student':
            # self.gamma = Parameter(torch.zeros(1)).cuda()
            self.conv_pam = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.conv_cam = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        else:
            self.conv_pam = Identity()
            self.conv_cam = Identity()

    def forward(self, x):
        out = self.conv_pam(self.PAM(x)) + self.conv_cam(self.CAM(x))

        return out

