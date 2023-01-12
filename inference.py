import os
import torch

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import time

from model.model_cnn_lstm_causal import ConvLSTM
from model.stft import StftHandler

from dataset import DatasetInf, DataLoader



def build_model(path_state):
    snapshot = torch.load(
        os.path.join(path_state, 'last_snapshot.tar'),
        map_location='cpu'
    )
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
        conv_kwargs=conv_kwargs, )

    state_dict = snapshot['model']
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model

def build_data(folder_path, batch_size):
    np.random.seed(77)
    torch.manual_seed(42)
    farend_path = os.path.join(folder_path, 'farend_speech')
    near_mic_path = os.path.join(folder_path, 'nearend_mic_signal')
    near_speech_path = os.path.join(folder_path, 'nearend_speech')


    test_dataset = DatasetInf(farend_path, near_mic_path, near_speech_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return test_loader


def processing_signal(farend, near_mic, model, t, device):
    # farend = torch.from_numpy(farend)
    # near_mic = torch.from_numpy(near_mic)

    # farend = farend.to(device, dtype=torch.float)
    # farend = farend.unsqueeze(0)

    # near_mic = near_mic.to(device, dtype=torch.float)
    # near_mic = near_mic.unsqueeze(0)

    first_step = 480
    frame = int(t * 16000 / 1000)

    current_frame_farend = farend[:, :first_step]
    rest_signal_farend = farend[:, first_step:]

    current_frame_mic = near_mic[:, :first_step]
    rest_signal_mic = near_mic[:, first_step:]

    #минимальная необходимая задержка 480 отчетов
    #то есть 30мс
    array_out = torch.zeros(480).to(device)
    print('processing signal by 1 sec')
    t_out = []
    while rest_signal_farend.shape[-1] > 0:

        frame_tmp_farend = rest_signal_farend[:, :frame]
        current_frame_farend = torch.cat((current_frame_farend, frame_tmp_farend), 1)

        frame_tmp_mic = rest_signal_mic[:, :frame]
        current_frame_mic = torch.cat((current_frame_mic, frame_tmp_mic), 1)

        start_time = time.time()
        current_pred = model(current_frame_farend, current_frame_mic)
        print("--- %s seconds ---" % (time.time() - start_time), frame_tmp_farend.shape)

        t_out.append(time.time() - start_time)
        array_out = torch.cat((array_out, current_pred[0][-frame:]), 0)

        rest_signal_farend = rest_signal_farend[:, frame:]
        rest_signal_mic = rest_signal_mic[:, frame:]
    print(np.mean(t_out))
    return array_out


def processing_1sec(farend, near_mic, model, device):
    farend = torch.from_numpy(farend)
    near_mic = torch.from_numpy(near_mic)

    farend = farend.to(device, dtype=torch.float)
    farend = farend.unsqueeze(0)

    near_mic = near_mic.to(device, dtype=torch.float)
    near_mic = near_mic.unsqueeze(0)

    first_step = 16000

    current_frame_farend = farend[:, :first_step]
    current_frame_mic = near_mic[:, :first_step]

    print('warm up')
    for i in range(2):
        current_pred = model(current_frame_farend, current_frame_mic)

    t_out = []
    print('processing 1 sec')
    for i in range(30):

        start_time = time.time()
        current_pred = model(current_frame_farend, current_frame_mic)
        print("--- %s seconds ---" % (time.time() - start_time), current_pred.shape)

        t_out.append(time.time() - start_time)
    print(np.mean(t_out))

def main():
    device = 'cpu'
    path_state = './results/cnn_lstm_causal'

    model = build_model(path_state)
    model = model.to(device)

    val_loader = build_data(folder_path='dataset-test-real', batch_size=1)
    farend, near_mic, target, path_id = next(iter(val_loader))


    # target = target[0].detach().cpu().numpy()


    #Проверка на скорость обработки одной секунды сигнала:
    #Обрабатывается только одна секунда сигнала, без записи
    #и без учета предыдущих
    farend_test = farend[0].detach().cpu().numpy()
    near_mic_test = near_mic[0].detach().cpu().numpy()
    test_1 = processing_1sec(farend_test, near_mic_test, model, device)



    #Предсказание на 6 рандомных сэмплах из датасета Real
    #Задержка 30мс
    #Сигнал обрабатывается каждую секунду, с учетом предыдущих секунд
    for farend, near_mic, target, path_id in tqdm(val_loader):
        farend = farend.to(device, dtype=torch.float)
        near_mic = near_mic.to(device, dtype=torch.float)

        pred = processing_signal(farend, near_mic, model, t=1000, device='cpu')
        sf.write(f'./dataset-test-real/pred_cnn/{path_id[0]}_pred.wav', pred.detach().cpu().numpy(), samplerate=16_000)



if __name__ == '__main__':
    main()