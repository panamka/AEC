import torch
import torch.nn as nn

class StftHandler(nn.Module):
    def __init__(self, n_fft=512, win_len=480, hop_len=160):
        super(StftHandler, self).__init__()
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        window = torch.hann_window(self.win_len)
        self.register_buffer('stft_window', window)

    def wave_to_spec(self, x):
        sps = torch.stft(x, n_fft=self.n_fft, win_length=self.win_len,
                         hop_length=self.hop_len,
                         window=self.stft_window,
                         return_complex=True)
        return sps

    def spec_to_mag(self, sps):
        mag = sps.abs() + 1e-8
        return mag

    def spec_to_wave(self, spec, length):
        wavform = torch.istft(
            spec,
            n_fft=self.n_fft, win_length=self.win_len,
            hop_length=self.hop_len,
            length=length
        )
        return wavform
