import torch

from dataset import AECDataset, DataLoader
from model.model_v1 import Conformer
from model.stft import StftHandler

conf_kwargs = dict(
    dim=256,
    dim_head=64,
    heads=4,
    ff_mult=2,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    attn_dropout=0.1,
    ff_dropout=0.1,
    conv_dropout=0.1,
    look_ahead=6,
)

model = Conformer(
    stft=StftHandler(),
    n_channels=16,
    num_layers=12,
    inp_dim=514,#257,
    out_dim=257,
    conformer_kwargs=conf_kwargs,)

farend_path = "./synthetic/farend_speech"
echo_path = "./synthetic/echo_signal"
near_mic_path = "./synthetic/nearend_mic_signal"
near_speech_path = "./synthetic/nearend_speech"
train_dataset = AECDataset(farend_path, echo_path, near_mic_path, near_speech_path)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

farend, near_mic, target = next(iter(train_loader))

device = 'cuda:0'


model = model.to(device)
farend = farend.to(device, dtype=torch.float)
near_mic = near_mic.to(device, dtype=torch.float)

model(farend, near_mic)

print(model.parameters())
