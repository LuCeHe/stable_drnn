import os

from GenericTools.stay_organized.download_utils import download_and_unzip

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

GSCDIR1 = os.path.abspath(os.path.join(CDIR, '..', 'data', 'GSC1'))
GSCDIR2 = os.path.abspath(os.path.join(CDIR, '..', 'data', 'GSC2'))
os.makedirs(GSCDIR1, exist_ok=True)
os.makedirs(GSCDIR2, exist_ok=True)

def download():
    if len(os.listdir(GSCDIR1)) == 0:
        data_links = ['http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz']
        download_and_unzip(data_links, GSCDIR1)

    if len(os.listdir(GSCDIR2)) == 0:
        data_links = ['http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz']
        download_and_unzip(data_links, GSCDIR2)


if __name__ == '__main__':
    download()
