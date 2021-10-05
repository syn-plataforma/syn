"""Download and extract file."""
import gzip
import os
import shutil
import sys
import zipfile
from pathlib import Path

import gensim.downloader as api
from six.moves import urllib

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.nlp.embeddings import get_pretrained_word_embeddings_model_name

load_environment_variables()
log = set_logger()


def download(url: str, local_dir: str):
    """Download and extract file."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    log.info(f"Downloading from {url} to: '{local_dir}'")
    filename = url.split('/')[-1]
    filepath = os.path.join(local_dir, filename)

    # Download.
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        statinfo = os.stat(filepath)
        log.info(f"Successfully downloaded {filename} {statinfo.st_size} bytes.")

    return filepath


def extract_zip(filepath: str, local_dir: str):
    # Extract.
    log.info(f"Extracting '{filepath}' to '{local_dir}'")
    zip_ref = zipfile.ZipFile(filepath, 'r')
    zip_ref.extractall(local_dir)
    zip_ref.close()
    os.remove(filepath)
    return zip_ref.namelist()


def extract_gzip(filepath: str, local_dir: str):
    # Extract.
    log.info(f"Extracting '{filepath}' to '{local_dir}'")
    extracted_file_path = filepath.split('.gz')[0]
    with gzip.open(filepath, 'rb') as f_in:
        with open(extracted_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(filepath)
    return extracted_file_path


def download_and_extract(url: str, local_dir: str):
    # Download.
    filepath = download(url=url, local_dir=local_dir)

    # Extract.
    if 'zip' == filepath.split('.')[-1]:
        return extract_zip(filepath=filepath, local_dir=local_dir)
    elif 'gz' == filepath.split('.')[-1]:
        return extract_gzip(filepath=filepath, local_dir=local_dir)


def download_gensim_vectors(model_name: str = 'glove-wiki-gigaword-100', local_dir: str = ''):
    for name in get_pretrained_word_embeddings_model_name(model_name):
        log.info(f"Downloading {name} ....")
        model = api.load(name)
        filename = Path(local_dir) / f"{name}.txt"
        model.save_word2vec_format(filename, binary=False)
        log.info(f"Downloaded and saved: '{name}'.")
