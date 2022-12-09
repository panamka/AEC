import glob
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random



#audio_files_utt = glob.glob(f'{path}/**/*.wav', recursive=True)

class AECDataset(Dataset):
    def __init__(self, root_farend, root_echo, root_near_mic, root_near_speech, train=True):
        self.audio_files_farend = glob.glob(f'{root_farend}/**/*.wav', recursive=True)
        self.audio_files_echo = glob.glob(f'{root_echo}/**/*.wav', recursive=True)
        self.audio_files_near_mic = glob.glob(f'{root_near_mic}/**/*.wav', recursive=True)
        self.audio_files_near_speach = glob.glob(f'{root_near_speech}/**/*.wav', recursive=True)
        self.max_len = 160000
        #self.transform = Transform
        #self.mask_gen = Masking()
        #np.random.RandomState(42).shuffle(self.audio_files_noises)
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
        #echo, _ = sf.read(echo_path, dtype='float32')
        near_mic, _ = sf.read(near_mic_path, dtype='float32')
        target, _ = sf.read(near_speech_path, dtype='float32')

        farend = farend
        near_mic = near_mic
        target = target
        return farend, near_mic, target

    def __getitem__(self, idx):
        farend_path = self.audio_files_farend[idx]
        #echo_path = self.audio_files_echo[idx]
        near_mic_path = self.audio_files_near_mic[idx]
        near_speech_path = self.audio_files_near_speach[idx]

        farend, near_mic, target = self.extractor(farend_path, near_mic_path, near_speech_path)

        if len(farend_path) < self.max_len:
            farend = self.padding(farend, self.max_len)
            #echo_path = self.padding(echo_path, self.max_len)
            near_mic = self.padding(near_mic, self.max_len)
            target = self.padding(target, self.max_len)

        return farend, near_mic, target


def main():
    farend_path = "./synthetic/farend_speech"
    echo_path = "./synthetic/echo_signal"
    near_mic_path = "./synthetic/nearend_mic_signal"
    near_speech_path = "./synthetic/nearend_speech"


    train_dataset = AECDataset(farend_path, echo_path, near_mic_path, near_speech_path)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(len(train_dataset), len(train_loader))
    farend, near_mic, target = next(iter(train_loader))

    print(farend.shape)
    #print(echo_path.shape)
    print(near_mic.shape)
    print(target.shape)

if __name__ == '__main__':
    main()