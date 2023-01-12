import torch
import torch.nn as nn
from torch.nn import functional as F

from .stft import StftHandler
from .norm import ChannelNorm, GroupChanNorm
from .act import GLU, Swish

from einops.layers.torch import Rearrange

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


class ConvLSTM(nn.Module):
    def __init__(self, stft: StftHandler,  num_layers, inp_dim, out_dim, conv_kwargs):
        super(ConvLSTM, self).__init__()
        self.stft = stft
        self.num_layers = num_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        dim = conv_kwargs['dim']
        encoder_dim = dim * 2
        casual = True
        expansion_factor = conv_kwargs['conv_expansion_factor']
        kernel_size = conv_kwargs['conv_kernel_size']
        dropout= conv_kwargs['conv_dropout']

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not casual else (kernel_size - 1, 0)

        self.farend_encoder = nn.Sequential(
            ChannelNorm(num_channels=inp_dim),
            nn.Conv1d(inp_dim,  inner_dim * 2, kernel_size=1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            GroupChanNorm(16, inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

        self.mixture_encoder = nn.Sequential(
            ChannelNorm(num_channels=inp_dim),
            nn.Conv1d(inp_dim,  inner_dim * 2, kernel_size=1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            GroupChanNorm(16, inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=dim*2, hidden_size=encoder_dim,
                            bidirectional=False, num_layers=num_layers, batch_first=True)
        self.final = nn.Linear(encoder_dim, out_dim)

    def forward(self, farend, mixture):
        spec_farend = self.stft.wave_to_spec(farend)
        mag_farend = self.stft.spec_to_mag(spec_farend)

        spec_mixture = self.stft.wave_to_spec(mixture)
        mag_mixture = self.stft.spec_to_mag(spec_mixture)

        farend_encoder_out = self.farend_encoder(mag_farend)
        mixture_encoder_out = self.mixture_encoder(mag_mixture)

        cat = torch.cat([farend_encoder_out, mixture_encoder_out], dim=-1)

        out, (h, c) = self.lstm(cat)

        mask = self.final(out)

        mask = mask.transpose(1, 2)

        spec_estimate = spec_mixture * mask
        wave_estimate = self.stft.spec_to_wave(spec_estimate, mixture.size(-1))

        return wave_estimate

