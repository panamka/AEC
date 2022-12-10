import glob
import soundfile as sf
import torch.nn.utils.rnn
from torch.utils.data import Dataset, DataLoader
import numpy as np


def collate_fn(data):
    data = [
        {'farend': farend,
         'nearend': nearend,
         'target': target,
         } for farend, nearend, target in data
    ]
    farend_audios = torch.nn.utils.rnn.pad_sequence([obj['farend'] for obj in data], batch_first=True,
                                                    padding_value=0.0)
    nearend_audios = torch.nn.utils.rnn.pad_sequence([obj['nearend'] for obj in data], batch_first=True,
                                                    padding_value=0.0)
    target_audios = torch.nn.utils.rnn.pad_sequence([obj['target'] for obj in data], batch_first=True,
                                                    padding_value=0.0)
    return farend_audios, nearend_audios, target_audios


class AECDataset(Dataset):
    def __init__(self, root_farend, root_echo, root_near_mic, root_near_speech, train=True):
        self.audio_files_farend = glob.glob(f'{root_farend}/**/*.wav', recursive=True)
        self.audio_files_echo = glob.glob(f'{root_echo}/**/*.wav', recursive=True)
        self.audio_files_near_mic = glob.glob(f'{root_near_mic}/**/*.wav', recursive=True)
        self.audio_files_near_speech = glob.glob(f'{root_near_speech}/**/*.wav', recursive=True)
        self.max_len = 160000
        train_len = int(0.8 * len(self.audio_files_farend))
        if train:
            self.audio_files_farend = self.audio_files_farend[:train_len]
        else:
            self.audio_files_farend = self.audio_files_farend[train_len:]

        print(f'Number of samples {len(self.audio_files_farend)}')

    def __len__(self):
        return len(self.audio_files_farend)


    def padding(self, signal, max_len):
        array_pad = np.zeros(max_len - len(signal))
        signal = np.append(signal, array_pad)
        return signal

    def extractor(self, farend_path, near_mic_path, near_speech_path):
        farend, sr = sf.read(farend_path, dtype='float32')
        near_mic, _ = sf.read(near_mic_path, dtype='float32')
        target, _ = sf.read(near_speech_path, dtype='float32')
        return farend, near_mic, target

    def __getitem__(self, idx):
        farend_path = self.audio_files_farend[idx]
        near_mic_path = self.audio_files_near_mic[idx]
        near_speech_path = self.audio_files_near_speech[idx]
        farend, near_mic, target = self.extractor(farend_path, near_mic_path, near_speech_path)

        if len(farend_path) < self.max_len:
            farend = self.padding(farend, self.max_len)
            near_mic = self.padding(near_mic, self.max_len)
            target = self.padding(target, self.max_len)
        return farend, near_mic, target


class DatasetInf(Dataset):
    def __init__(self, root_farend, root_near_mic, root_near_speech):
        self.audio_files_farend = glob.glob(f'{root_farend}/**/*.wav', recursive=True)
        self.audio_files_near_mic = glob.glob(f'{root_near_mic}/**/*.wav', recursive=True)
        self.audio_files_near_speech = glob.glob(f'{root_near_speech}/**/*.wav', recursive=True)
        print(f'Number of samples farend {len(self.audio_files_farend)}')
        print(f'Number of samples mic {len(self.audio_files_near_mic)}')
        print(f'Number of samples speech {len(self.audio_files_near_speech)}')
    def __len__(self):
        return len(self.audio_files_farend)

    def extractor(self, farend_path, near_mic_path, near_speech_path):
        path_id_splitted = farend_path.split('_')
        if len(path_id_splitted) < 7:
            path_id = path_id_splitted[2]
        else:
            path_id = '_'.join((path_id_splitted[2], '_'))
        farend, sr = sf.read(farend_path, dtype='float32')
        near_mic, _ = sf.read(near_mic_path, dtype='float32')
        target, _ = sf.read(near_speech_path, dtype='float32')

        max_len = min(len(farend), len(near_mic))
        farend = farend[:max_len]
        near_mic = near_mic[:max_len]
        return farend, near_mic, target, path_id

    def __getitem__(self, idx):
        farend_path = self.audio_files_farend[idx]
        near_mic_path = self.audio_files_near_mic[idx]
        near_speech_path = self.audio_files_near_speech[idx]
        farend, near_mic, target, path_id = self.extractor(farend_path, near_mic_path, near_speech_path)
        return farend, near_mic, target, path_id



def main():
    np.random.seed(77)
    torch.manual_seed(42)
    farend_path = "dataset_synthetic/farend_speech"
    echo_path = "dataset_synthetic/echo_signal"
    near_mic_path = "dataset_synthetic/nearend_mic_signal"
    near_speech_path = "dataset_synthetic/nearend_speech"



    batch_size = 3
    train_dataset = AECDataset(farend_path, echo_path, near_mic_path, near_speech_path)
    test_dataset = AECDataset(farend_path, echo_path, near_mic_path, near_speech_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    farend, near_mic, target = next(iter(train_loader))
    print(farend.shape)
    # farend, near_mic, target = next(iter(test_loader))


if __name__ == '__main__':
    main()