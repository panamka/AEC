import torch
import torch.nn as nn
from torch.nn import functional as F

from .stft import StftHandler
from .aec_filters import FDAF, FDKF


class NAECwithFDAF(nn.Module):
    def __init__(self, stft: StftHandler, input_size, hidden_size, num_layers, aec_filter: FDAF):
        super(NAECwithFDAF, self).__init__()
        self.stft = stft
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.aec_filter = aec_filter

        self.lstm = nn.LSTM(input_size=input_size*2, hidden_size=hidden_size,
                            bidirectional=False, num_layers=num_layers, batch_first=True)
        self.fc_mask = nn.Sequential(
                                    nn.Linear(hidden_size, input_size),
                                    nn.Sigmoid())
        self.fc_mu = nn.Sequential(
                                    nn.Linear(hidden_size, input_size),
                                    nn.Sigmoid())

    def forward(self, farend_x, input_y):
        #calculating a mask for a reference signal (far-end signal Y)
        #multiply mask*farend_x and use it in adaptive filter
        #input_y without changing

        spec_farend = self.stft.wave_to_spec(farend_x)
        mag_farend = self.stft.spec_to_mag(spec_farend)

        spec_input = self.stft.wave_to_spec(input_y)
        mag_input = self.stft.spec_to_mag(spec_input)

        cat = torch.cat([mag_farend, mag_input], dim=-2)
        cat = cat.transpose(1, 2)
        out, (h, c) = self.lstm(cat)
        mask = self.fc_mask(out)

        mu = self.fc_mu(out)
        mask = mask.transpose(1, 2)

        spec_estimate = spec_farend * mask
        wave_estimate = self.stft.spec_to_wave(spec_estimate, farend_x.size(-1))

        out_aec = self.aec_filter.process_hop(input_y, wave_estimate, mu)

        return out_aec

class NAECwithFDKF(nn.Module):
    def __init__(self, stft: StftHandler, input_size, hidden_size, num_layers, aec_filter: FDKF):
        super(NAECwithFDKF, self).__init__()
        self.stft = stft
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.aec_filter = aec_filter

        self.lstm = nn.LSTM(input_size=input_size*2, hidden_size=hidden_size,
                            bidirectional=False, num_layers=num_layers, batch_first=True)
        self.fc_mask = nn.Sequential(
                                    nn.Linear(hidden_size, input_size),
                                    nn.Sigmoid())
        # self.fc_mu = nn.Sequential(
        #                             nn.Linear(hidden_size, input_size),
        #                             nn.Sigmoid())

    def forward(self, farend_x, input_y):
        #calculating a mask for a reference signal (far-end signal Y)
        #multiply mask*farend_x and use it in adaptive filter
        #input_y without changing

        spec_farend = self.stft.wave_to_spec(farend_x)
        mag_farend = self.stft.spec_to_mag(spec_farend)

        spec_input = self.stft.wave_to_spec(input_y)
        mag_input = self.stft.spec_to_mag(spec_input)

        cat = torch.cat([mag_farend, mag_input], dim=-2)
        cat = cat.transpose(1, 2)
        out, (h, c) = self.lstm(cat)
        mask = self.fc_mask(out)

        # mu = self.fc_mu(out)
        mask = mask.transpose(1, 2)

        spec_estimate = spec_farend * mask
        wave_estimate = self.stft.spec_to_wave(spec_estimate, farend_x.size(-1))


        out_aec = self.aec_filter.process_hop(input_y, wave_estimate)
        print(out_aec.shape)

        return out_aec

