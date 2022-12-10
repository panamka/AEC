import torch
import os
import numpy as np
from tqdm.auto import tqdm
import soundfile as sf

from model.model_v1 import Conformer
from model.stft import StftHandler

from dataset import DataLoader, DatasetInf

def build_model(path):
    '''
    :param path: path to the folder with snapshot
    :return: trained model in eval mode
    '''
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
    )

    model = Conformer(
        stft=StftHandler(),
        num_layers=8,
        inp_dim=257,  # 257,
        out_dim=257,
        conformer_kwargs=conf_kwargs, )


    snapshot = torch.load(
        os.path.join(path, 'last_snapshot.tar'),
        map_location='cpu'
    )
    state_dict = snapshot['model']
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model

def load_data(batch_size=1):
    np.random.seed(77)
    torch.manual_seed(42)
    farend_path = "dataset-test-real/farend_speech"
    near_mic_path = "dataset-test-real/nearend_mic_signal"
    near_speech_path = "dataset-test-real/nearend_speech"

    test_dataset = DatasetInf(farend_path, near_mic_path, near_speech_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return test_loader


def main():
    device='cuda:0'

    #path to the folder with last_snapshot.tar
    path = './results'
    model = build_model(path)
    model.to(device)

    #eval on several samples from real dataset AEC Challenge
    test_loader = load_data()
    farend, near_mic, target, path_id = next(iter(test_loader))

    farend = farend.to(device, dtype=torch.float)
    near_mic = near_mic.to(device, dtype=torch.float)


    for farend, near_mic, target, path_id in tqdm(test_loader):
        farend = farend.to(device, dtype=torch.float)
        near_mic = near_mic.to(device, dtype=torch.float)
        #target = target.to(device, dtype=torch.float)
        pred = model(farend, near_mic)
        sf.write(f'./dataset-test-real/pred/{path_id[0]}_pred.wav', pred[0].detach().cpu().numpy(), samplerate=16_000)


if __name__ == '__main__':
    main()
