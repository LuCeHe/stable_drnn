import os

from GenericTools.stay_organized.download_utils import download_and_unzip

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

SOLIDIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'Soli'))
os.makedirs(SOLIDIR, exist_ok=True)


def download():

    src = os.path.join(SOLIDIR, 'download')
    dst = os.path.join(SOLIDIR, 'SoliData.zip')
    if not os.path.exists(dst):
        data_links = ['https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT/download']
        download_and_unzip(data_links, SOLIDIR)
        os.rename(src, dst)

    from zipfile import ZipFile
    with ZipFile(dst, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(dst.replace('.zip', ''))


if __name__ == '__main__':
    download()
