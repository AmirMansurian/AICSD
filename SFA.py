
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class SFA(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, ks=7, groups=16, gamma=2, b=1):
        super(SFA, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma)) + 2
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        p = ks // 2
        c = channel // (groups * 2)
        self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=p, groups=c, bias=False)

        self.sig = nn.Sigmoid()

    def forward(self, x, is_student: bool = True):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # -------- Channel attention --------
        b1, c1, h, w = x_0.shape
        y = (self.avg_pool(x_0) * 1.15 + self.max_pool(x_0) * 0.25).view([b1, 1, c1])

        if is_student:
            # use learnable conv1
            y = self.conv1(y)
            y = self.sig(y).view([b1, c1, 1, 1])
        else:
            # fallback: just normalize and sigmoid
            y = self.sig(y).view([b1, c1, 1, 1])

        xn = x_0 * y

        # -------- Spatial attention --------
        b2, c2, h, w = x_1.shape
        x_h = torch.mean(x_1, dim=3, keepdim=True).view(b2, c2, h)
        x_w = torch.mean(x_1, dim=2, keepdim=True).view(b2, c2, w)

        if is_student:
            # use learnable conv2
            x_h = self.sig(self.conv2(x_h)).view(b2, c2, h, 1)
            x_w = self.sig(self.conv2(x_w)).view(b2, c2, 1, w)
        else:
            # fallback: only sigmoid without conv
            x_h = self.sig(x_h).view(b2, c2, h, 1)
            x_w = self.sig(x_w).view(b2, c2, 1, w)

        xs = x_1 * x_h * x_w

        # -------- Combine --------
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        return out
