import torch
import torch.nn as nn
from torch.nn import functional as F

from .stft import StftHandler
from .block import ConformerBlock, ConformerBlockStream
from .norm import ChannelNorm

class Transposer(nn.Module):
    def __init__(self):
        super(Transposer, self).__init__()

    def forward(self, x):
        return x.transpose(1, 2)

class Conformer(nn.Module):
    def __init__(self, stft: StftHandler,  num_layers, inp_dim, out_dim, conformer_kwargs):
        super(Conformer, self).__init__()
        self.stft = stft
        self.num_layers = num_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.conformer_kwargs = conformer_kwargs
        encoder_dim = conformer_kwargs['dim']

        #self.proj = nn.Conv1d(inp_dim, conformer_kwargs['dim'], kernel_size=1)

        farend_encoder = [ConformerBlockStream(**conformer_kwargs) for _ in range(2)]
        self.farend_encoder = nn.Sequential(
            ChannelNorm(num_channels=inp_dim),
            nn.Conv1d(inp_dim, conformer_kwargs['dim'], kernel_size=1),
            Transposer(),
            *farend_encoder
            )

        mixture_encoder = [ConformerBlockStream(**conformer_kwargs) for _ in range(2)]
        self.mixture_encoder = nn.Sequential(
            ChannelNorm(num_channels=inp_dim),
            nn.Conv1d(inp_dim, conformer_kwargs['dim'], kernel_size=1),
            Transposer(),
            *mixture_encoder
            )

        self.channel_norm = ChannelNorm(num_channels=conformer_kwargs['dim'] * 2)
        self.proj_cat = nn.Linear(conformer_kwargs['dim'] * 2, conformer_kwargs['dim'], bias=False)
        conformer_blocks = [ConformerBlockStream(**conformer_kwargs) for _ in range(num_layers)]

        self.conformer = nn.Sequential(*conformer_blocks)
        self.final = nn.Linear(encoder_dim, out_dim)

    def forward(self, farend, mixture):
        spec_farend = self.stft.wave_to_spec(farend)
        mag_farend = self.stft.spec_to_mag(spec_farend)
        spec_mixture = self.stft.wave_to_spec(mixture)
        mag_mixture = self.stft.spec_to_mag(spec_mixture)

        farend_encoder_out = self.farend_encoder(mag_farend)
        mixture_encoder_out = self.mixture_encoder(mag_mixture)

        cat = torch.cat([farend_encoder_out, mixture_encoder_out], dim=-1)


        cat = self.proj_cat(cat)

        out = self.conformer(cat)
        mask = self.final(out)

        mask = mask.transpose(1, 2)

        spec_estimate = spec_mixture * mask
        wave_estimate = self.stft.spec_to_wave(spec_estimate, mixture.size(-1))


        return wave_estimate

