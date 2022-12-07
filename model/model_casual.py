import torch
from torch import nn
from torch.nn import functional as F

from .stft import StftHandler
from .block import ConformerBlockStream
from .norm import ChannelNorm

def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parametes()])
    return neles / 10**6 if Mb else neles

class ConformerVad(nn.Module):
    def __init__(self, stft: StftHandler, num_layers, inp_dim, out_dim, conformer_kwargs):
        super(ConformerVad, self).__init__()
        self.stft = stft
        self.num_layers = num_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.conformer_kwargs = conformer_kwargs
        encoder_dim = conformer_kwargs['dim']
        self.channel_norm = ChannelNorm(num_channels=inp_dim)
        self.proj = nn.Conv1d(inp_dim, conformer_kwargs['dim'], kernel_size=1)

        conformer_blocks = [ConformerBlockStream(**conformer_kwargs) for _ in range(num_layers)]
        self.conformer = nn.Sequential(*conformer_blocks)

        self.final = nn.Linear(encoder_dim, 1)
        self.mask = nn.Sigmoid()

    def forward(self, mixture):
        spec = self.stft.wave_to_spec(mixture)
        mag = self.stft.spec_to_mag(spec)

        out = self.channel_norm(mag)
        out = self.proj(out)
        out = out.transpose(1, 2)
        out = self.conformer(out)
        vad_logits = self.final(out)

        pred_vad_mask = torch.sigmoid(vad_logits)
        pred_vad_mask = pred_vad_mask.transpose(1, 2)

        pred_vad_mask = F.interpolate(
            pred_vad_mask, size=mixture.shape[-1], mode='linear',
        ).squeeze(1)

        return pred_vad_mask