import torch
import torch.nn as nn
from torch.nn import Parameter


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        self.sigmoid = nn.Sigmoid()
        self.w = Parameter(torch.Tensor(1, channel, 1, 1))
        self.b = Parameter(torch.Tensor(1, channel, 1, 1))
        self.w.data.fill_(0.5)
        self.b.data.fill_(-0.5)

    def forward(self, x, is_student = False):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        if is_student:
            # Learned parameters For the student model
            y = self.fc(y).reshape(b, c, 1, 1)
            sub1 = y * self.w + self.b
            y = self.sigmoid(y) * self.sigmoid(sub1)
        else:
            # Fixed parameters For the teacher model
            y = y.reshape(b, c, 1, 1)
            y = y * self.sigmoid(y)
            
        return x * y