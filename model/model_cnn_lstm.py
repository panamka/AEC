import torch
import torch.nn as nn
from torch.nn import functional as F

from .stft import StftHandler
from .block import ConformerBlock, ConformerBlockStream, BlockStream
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
        encoder_dim = 512
        n_channels = 16


        self.farend_encoder = nn.Sequential(nn.Conv2d(1, out_channels=n_channels, kernel_size=(3,1), stride=(1,1)),
                                            nn.BatchNorm2d(num_features=n_channels),
                                            nn.ReLU(inplace=True),

                                            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 1), stride=(1,1)),
                                            nn.BatchNorm2d(num_features=n_channels),
                                            nn.ReLU(inplace=True)
                                            )

        self.mixture_encoder = nn.Sequential(nn.Conv2d(1, out_channels=n_channels,  kernel_size=(3, 1), stride=(1,1)),
                                             nn.BatchNorm2d(num_features=n_channels),
                                             nn.ReLU(inplace=True),

                                             nn.Conv2d(in_channels=n_channels, out_channels=n_channels,  kernel_size=(3, 1), stride=(1,1)),
                                             nn.BatchNorm2d(num_features=n_channels),
                                             nn.ReLU(inplace=True)
                                             )

        input_size = 16 * 506
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=encoder_dim,
                            bidirectional=False, num_layers=num_layers, batch_first=True)
        self.final = nn.Linear(encoder_dim, out_dim)

    def forward(self, farend, mixture):
        spec_farend = self.stft.wave_to_spec(farend)
        mag_farend = self.stft.spec_to_mag(spec_farend)

        spec_mixture = self.stft.wave_to_spec(mixture)
        mag_mixture = self.stft.spec_to_mag(spec_mixture)

        mag_farend = mag_farend.unsqueeze(1)
        mag_mixture = mag_mixture.unsqueeze(1)

        farend_encoder_out = self.farend_encoder(mag_farend)
        mixture_encoder_out = self.farend_encoder(mag_mixture)



        cat = torch.cat([farend_encoder_out, mixture_encoder_out], dim=-2)


        sizes = cat.size()

        # [Batch. n_channels, freatures, num_timesteps] -> [Batch, features, num_timesteps]
        cat = cat.view(sizes[0], sizes[1] * sizes[2], sizes[3])

        cat = cat.transpose(1, 2)
        #[Batch. num_timesteps, freatures]
        #print(cat.shape)

        out, (h, c) = self.lstm(cat)

        mask = self.final(out)

        mask = mask.transpose(1, 2)

        spec_estimate = spec_mixture * mask
        wave_estimate = self.stft.spec_to_wave(spec_estimate, mixture.size(-1))


        return wave_estimate

