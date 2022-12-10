import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import os
from shutil import rmtree
import math
#from TensorboardLogger import TensorboardLogger
import soundfile as sf

from dataset import AECDataset, DataLoader, collate_fn
from model.model_v1 import Conformer
from model.stft import StftHandler

from metrics.aec_metrics import calc_stoi as stoi
from metrics.aec_metrics import calc_sdr as sdr
def to_numpy(x):
    return x.detach().cpu().numpy()

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32 -1))


def calc_stoy(x, y):
    stoi
def train_epoch(model, farend, near_mic, target, optimizer, criterion, metric_dict,  max_norm_grad, device):
    log = {}
    losses = []

    farend = farend.to(device, dtype=torch.float)
    near_mic = near_mic.to(device, dtype=torch.float)

    target = target.to(device, dtype=torch.float)

    optimizer.zero_grad()
    for param_group in optimizer.param_groups:
        log['lr_train'] = param_group['lr']

    pred = model(farend, near_mic)
    loss = criterion(pred, target)

    #loss = loss.mean()
    loss.backward()
    grad_norm = clip_grad_norm_(
        model.parameters(), max_norm_grad).item(
    )
    log['grad_norm'] = grad_norm
    optimizer.step()
    #scheduler.step()
    losses.append(loss.item())
    #log['train_loss'] = np.mean(losses)
    for name, metrics in metric_dict.items():

        values = metrics(pred, target)
        values = to_numpy(values)
        log[name] = np.mean(values)
    print(np.mean(losses))

    # sf.write(f'./tmp/pred.wav', pred[0].detach().cpu().numpy(), samplerate=16_000)
    # sf.write(f'./tmp/farend.wav', farend[0].detach().cpu().numpy(), samplerate=16_000)
    # sf.write(f'./tmp/near_mic.wav', near_mic[0].detach().cpu().numpy(), samplerate=16_000)
    # sf.write(f'./tmp/target.wav', target[0].detach().cpu().numpy(), samplerate=16_000)

    return {'train_loss' : np.mean(losses)} | log


def main():
    train_history = defaultdict(list)
    max_grad_norm = 5
    device = 'cuda:0'

    n_epoch = 10
    batch_size = 1
    lr = 1e-3
    start_epoch = 0

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
        num_layers=2,
        inp_dim=257,  # 257,
        out_dim=257,
        conformer_kwargs=conf_kwargs, )

    metric_dict = {
        "STOI" : stoi,
        "SDR" : sdr,
    }

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.L1Loss()
    saveroot = './results/'

    farend_path = "dataset_synthetic/farend_speech"
    echo_path = "dataset_synthetic/echo_signal"
    near_mic_path = "dataset_synthetic/nearend_mic_signal"
    near_speech_path = "dataset_synthetic/nearend_speech"
    train_dataset = AECDataset(farend_path, echo_path, near_mic_path, near_speech_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    farend, near_mic, target = next(iter(train_loader))

    for epoch in tqdm(range(start_epoch, n_epoch + start_epoch)):
        train_dict = train_epoch(model, farend, near_mic, target, optimizer, criterion, metric_dict,  max_grad_norm, device)
        print('train metrics')
        for key, value in train_dict.items():
            #logger_tb.log(epoch, value, key, 'train')
            print(f'{key}: {value}')
            train_history[key].append(value)

        snapshot = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            #'sch': lr_scheduler.state_dict(),
            'train_history': train_history,
            #'val_history': val_history,
        }

        last_snapshot_path = os.path.join(saveroot, 'last_snapshot.tar')
        torch.save(snapshot, last_snapshot_path)


if __name__ == '__main__':
    main()