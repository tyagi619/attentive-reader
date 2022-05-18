'''
This script downloads the SQuAD 2.0 dataset,
GLoVe 300D word vectors and also preprocesses the
dataset.
1- Download GLoVe 300D word vectors
2- Download SQuAD 2.0
3- Pre-process the dataset and word vectors
'''

# TODO - import required libraries
from urllib.request import urlretrieve
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
from subprocess import run


def _get_filename_from_url(download_url):
    filename = download_url.split('/')[-1]
    return filename


def _download(download_url, save_path):
    # TODO - raise error if download fails

    # define a custom class inheriting tqdm to display
    # progress bar while downloading file
    class DownloadProgressBar(tqdm):
        def update_bar(self, n_blocks, b_size, t_size=None):
            if t_size is not None:
                self.total = t_size
            self.update(n_blocks * b_size - self.n)

    with DownloadProgressBar(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=download_url.split('/')[-1]) as t:
        filename, headers = urlretrieve(url=download_url,
                                        filename=save_path,
                                        reporthook=t.update_bar)

    return None


def download(download_url_list, output_dir):
    for name, download_url in download_url_list:
        filename = _get_filename_from_url(download_url)
        filePath = output_dir/filename

        if not filePath.exists():
            print(f'Downloading {name}...')
            _download(filePath, str(filePath))

        if filePath.exists() and (filePath.suffix=='.zip'):
            extracted_filename = filename.replace('.zip','')
            extracted_filePath = output_dir/extracted_filename
            if not extracted_filePath.exists():
                print(f'Extracting {filename}...')
                with ZipFile(str(filePath)) as zip_file:
                    zip_file.extractall(str(extracted_filePath))

    # download spacy english language model (only tokenizer is required)
    print(f'Download English language model for spacy...')
    run(['python3','-m','spacy','download','en'])

    return None


def pre_process(output_dir):
    # TODO - 
    pass


if __name__=='__main__':
    
    # TODO - download SQuAD and GLoVe

    # TODO - preprocess and save SQuAD and GLoVe

    pass