import torch
import torch.nn as nn
from torch.nn import functional as F

from .stft import StftHandler
from .block import ConformerBlockStream
from .norm import ChannelNorm

class Conformer(nn.Module):
    def __init__(self, stft: StftHandler, n_channels, num_layers, inp_dim, out_dim, conformer_kwargs):
        super(Conformer, self).__init__()
        self.stft = stft
        self.n_channels = n_channels
        self.conv_block_1 = nn.Sequential(nn.Conv2d(1, out_channels=n_channels, kernel_size=(3, 4), stride=(2, 1)),
                                          nn.BatchNorm2d(num_features=n_channels),
                                          nn.ReLU(inplace=True),)
        self.conv_block_2 = nn.Sequential(nn.Conv2d(1, out_channels=n_channels, kernel_size=(3, 4), stride=(2, 1)),
                                          nn.BatchNorm2d(num_features=n_channels),
                                          nn.ReLU(inplace=True),)
        self.num_layers = num_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.conformer_kwargs = conformer_kwargs
        encoder_dim = conformer_kwargs['dim']
        self.channel_norm = ChannelNorm(num_channels=inp_dim)
        self.proj = nn.Conv1d(inp_dim, conformer_kwargs['dim'], kernel_size=1)

        conformer_blocks = [ConformerBlockStream(**conformer_kwargs) for _ in range(num_layers)]
        self.conformer = nn.Sequential(*conformer_blocks)

        self.final = nn.Linear(encoder_dim, out_dim)
        self.mask = nn.Sigmoid()

    def forward(self, farend, mixture):
        spec_farend = self.stft.wave_to_spec(farend)
        # B, n_fft, t_steps (10, 257, 1001)
        mag_farend = self.stft.spec_to_mag(spec_farend)

        # B, 1,  n_fft, t_steps
        #mag_farend = mag_farend.unsqueeze(1)
        # B, n_channels,  n_fft_new, t_steps_new (10, 16, 128, 998)
       # mag_farend = self.conv_block_1(mag_farend)



        spec = self.stft.wave_to_spec(mixture)
        mag = self.stft.spec_to_mag(spec)
        #mag = mag.unsqueeze(1)
        #mag = self.conv_block_2(mag)

        # B, n_channels,  2xn_fft_new, t_steps_new (10, 16, 256, 998)
        #out = torch.cat((mag_farend, mag), 2)

        out = torch.cat((mag_farend, mag), 1)
        #print(out.shape)



        out = self.channel_norm(out)
        out = self.proj(out)
        out = out.transpose(1, 2)

        out = self.conformer(out)
        #print(out.shape)

        logits = self.final(out)
        mask = self.mask(logits)

        mask = mask.transpose(1, 2)
        #print(mask.shape, 'mask shape')

        spec_estimate = spec * mask
        wave_estimate = self.stft.spec_to_wave(spec_estimate, mixture.size(-1))

        #return spec_estimate, wave_estimate

        return wave_estimate

