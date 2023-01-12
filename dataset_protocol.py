import glob
import shutil
from tqdm.auto import tqdm

path = 'dataset-test-real'

audio_test = glob.glob(f'{path}/**/*.wav', recursive=True)
print(audio_test)


def copy_wav(path, dst, dub=False):
    splitted = path.split('_')
    joined = '_'.join(splitted[1:])
    path_new = '_'.join([dst, joined])
    shutil.copy(path, path_new)
    if dub == True:
        splitted = path_new.split('.')
        print(splitted)
        path_new = '_'.join(splitted)
        shutil.copy(path, path_new)


for path in tqdm(audio_test):
    if "doubletalk_lpb" in path:
        dst = 'dataset-test-real/farend_speech/'
        copy_wav(path, dst)
    elif "doubletalk_with_movement_lpb" in path:
        dst = 'dataset-test-real/farend_speech/'
        copy_wav(path, dst)
    elif 'doubletalk_mic' in path:
        dst = 'dataset-test-real/nearend_mic_signal/'
        copy_wav(path, dst)
    elif 'doubletalk_with_movement_mic' in path:
        dst = 'dataset-test-real/nearend_mic_signal/'
        copy_wav(path, dst)
    elif 'nearend_singletalk' in path:
        dst = 'dataset-test-real/nearend_speech/'
        copy_wav(path, dst)
        copy_wav(path, dst, dub=True)



