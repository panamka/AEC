from einops.layers.torch import Rearrange

import torch.nn as nn
import torch.nn.functional as F
from .norm import GroupChanNorm
from .act import GLU, Swish

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super(DepthWiseConv1d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(self,
                 dim,
                 casual=False,
                 expansion_factor=2,
                 kernel_size=31,
                 dropout=0.):
        super(ConformerConvModule, self).__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not casual else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            GroupChanNorm(16, inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)