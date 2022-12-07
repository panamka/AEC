import torch
import torch.nn as nn

class GroupChanNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        self.num_channels = num_channels

    def forward(self, x):

        #B,C,T
        B, C, T = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B * T, self.num_channels)

        x = self.norm(x)

        x = x.view(B, T, self.num_channels)
        x = x.permute(0, 2, 1).contiguous()

        return x

class ChannelNorm(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-8, affine=True):
        super().__init__(normalized_shape=num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        assert x.dim() == 3
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x

def layernorm_tf(x):
    mean = torch.mean(x, (1, 2), keepdim=True)
    var = torch.sqrt(torch.mean((x - mean) ** 2, (1, 2), keepdim=True) + 1e-8)

    y = (x - mean) / var

    return y

def layernorm_f(x: torch.Tensor):
    mean = torch.mean(x, (1), keepdim=True)
    var = torch.sqrt(torch.mean((x - mean) ** 2, 1, keepdim=True) + 1e-8)
    y = (x - mean) / var

    return y


def layernorm_t(x: torch.Tensor):
    mean = torch.mean(x, 2, keepdim=True)
    var = torch.sqrt(torch.mean((x - mean) ** 2, (2), keepdim=True) + 1e-8)
    y = (x - mean) / var

    return y

def lsms(x):
    x = torch.log10(x)
    mean = torch.mean(x , (2), keepdim=True)
    x = x - mean
    return x
