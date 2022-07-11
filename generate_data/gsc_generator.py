import os
from tqdm import tqdm
from GenericTools.stay_organized.download_utils import download_and_unzip

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

GSCDIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'GSC'))
os.makedirs(GSCDIR, exist_ok=True)


def download():
    if len(os.listdir(GSCDIR)) == 0:
        # data_links = ['http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz']
        data_links = ['http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz']
        download_and_unzip(data_links, GSCDIR)

    # read backgorundnoises, and assign 10% to test and validation, and the rest to train

    silences_path = os.path.join(GSCDIR, '_background_noise_')
    silences = os.listdir(silences_path)
    print(silences)
    silences = sorted(silences)

    print(silences)

    # make test numpys
    test_paths = os.path.join(GSCDIR, 'testing_list.txt')
    with open(test_paths) as f:
        test_files = f.readlines()
    # print(lines)

    audios = []

    import numpy as np
    from scipy.io.wavfile import read

    for file in tqdm(test_files):
        d = os.path.join(GSCDIR, *file.replace('\n', '').split('/'))
        fs, a = read(d)
        normal_a = (a-np.mean(a))/np.std(a)
        a = np.pad(normal_a, int((16000-len(a))/2)+1, mode='mean')[:16000]
        audios.append(a[None])

    audios = np.concatenate(audios)
    print(audios.shape)
    print(audios[0])
    np.random.shuffle(audios)
    print(audios.shape)
    print(audios[0])

def normalize_sound(img, label):
    print(img.shape)
    img = tf.cast(img, tf.float32) / 255.
    img = tf.image.resize(img,[180,180])
    return (img, label)

if __name__ == '__main__':
    # download()
    import tensorflow.compat.v2 as tf
    import tensorflow_datasets as tfds

    # Construct a tf.data.Dataset
    ds = tfds.load('speech_commands', split='test', shuffle_files=True, data_dir=GSCDIR)

    # Build your input pipeline
    ds = ds.shuffle(1024).map(normalize_sound).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    for example in ds.take(1):
        image, label = example["audio"], example["label"]
        print(image)
        print(label)
