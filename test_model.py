import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np

from model.model_cnn_lstm_causal import ConvLSTM
from model.stft import StftHandler

# conf_kwargs = dict(
#     dim=256,
#     dim_head=64,
#     heads=4,
#     ff_mult=2,
#     conv_expansion_factor=2,
#     conv_kernel_size=31,
#     attn_dropout=0.1,
#     ff_dropout=0.1,
#     conv_dropout=0.1,
#     look_ahead=6,
# )
#
# model = Conformer(
#     stft=StftHandler(),
#     num_layers=8,
#     inp_dim=257,  # 257,
#     out_dim=257,
#     conformer_kwargs=conf_kwargs, )

conv_kwargs = dict(
    dim=256,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    conv_dropout=0.1,
)

model = ConvLSTM(
    stft=StftHandler(),
    num_layers=4,
    inp_dim=257,
    out_dim=257,
    conv_kwargs=conv_kwargs,)

def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


device = 'cuda:0'
model.to(device)
print(param(model))

x = torch.rand(1, 16000).to(device, dtype=torch.float)
y = torch.rand(1, 16000).to(device, dtype=torch.float)

out = model(x, y)
