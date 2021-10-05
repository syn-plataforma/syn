"""
Downloads the following:
- Stanford CoreNLP
- Stanford Shift-Reduce Constituency Parser
- MongoDB Java Driver
- Glove vectors
- Word2Vec vectors
- FastText vectors

"""

import argparse
import os
import shutil
from pathlib import Path

from definitions import ROOT_DIR
from syn.helpers.download import download, download_and_extract, download_gensim_vectors
from syn.helpers.logging import set_logger

# Defines logger.
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        description='Download Stanford CoreNLP and Stanford Shift-Reduce Constituency Parser libraries, '
                    'MongoDB Java Driver and Glove vectors'
    )
    # download_corenlp
    parser.add_argument(
        '--download_corenlp',
        default=True,
        dest='download_corenlp',
        action='store_true',
        help="Download Stanford CoreNLP library."
    )
    # no_download_corenlp
    parser.add_argument(
        '--no_download_corenlp',
        dest='download_corenlp',
        action='store_false',
        help="Don't download Stanford CoreNLP library.."
    )
    # download_srparser
    parser.add_argument(
        '--download_srparser',
        default=True,
        dest='download_srparser',
        action='store_true',
        help="Download Stanford Shift-Reduce Constituency Parser library."
    )
    # no_download_srparser
    parser.add_argument(
        '--no_download_srparser',
        dest='download_srparser',
        action='store_false',
        help="Don't download Stanford Shift-Reduce Constituency Parser library."
    )
    # download_mongodb
    parser.add_argument(
        '--download_mongodb',
        default=True,
        dest='download_mongodb',
        action='store_true',
        help="Download MongoDB Java Driver."
    )
    # no_download_mongodb
    parser.add_argument(
        '--no_download_mongodb',
        dest='download_mongodb',
        action='store_false',
        help="Don't download MongoDB Java Driver."
    )
    # download_glove
    parser.add_argument(
        '--download_glove',
        default=True,
        dest='download_glove',
        action='store_true',
        help="Download Glove vectors."
    )
    # no_download_glove
    parser.add_argument(
        '--no_download_glove',
        dest='download_glove',
        action='store_false',
        help="Don't download Glove vectors."
    )
    # download_word2vec
    parser.add_argument(
        '--download_word2vec',
        default=True,
        dest='download_word2vec',
        action='store_true',
        help="Download Word2Vec vectors."
    )
    # no_download_word2vec
    parser.add_argument(
        '--no_download_word2vec',
        dest='download_word2vec',
        action='store_false',
        help="Don't download Word2Vec vectors."
    )
    # download_fasttext
    parser.add_argument(
        '--download_fasttext',
        default=True,
        dest='download_fasttext',
        action='store_true',
        help="Download Glove vectors."
    )
    # no_download_fasttext
    parser.add_argument(
        '--no_download_fasttext',
        dest='download_fasttext',
        action='store_false',
        help="Don't download Fasttext vectors."
    )
    args = parser.parse_args()

    return {
        'download_corenlp': args.download_corenlp,
        'download_srparser': args.download_srparser,
        'download_mongodb': args.download_mongodb,
        'download_glove': args.download_glove,
        'download_word2vec': args.download_word2vec,
        'download_fasttext': args.download_fasttext
    }


if __name__ == '__main__':
    # Input parameters.
    input_params = get_input_params()

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Download and extract Stanford CoreNLP.
    stanford_dir = Path(ROOT_DIR) / 'lib' / 'stanford'
    if input_params['download_corenlp']:
        stanford_corenlp_latest_url = 'https://nlp.stanford.edu/software/stanford-corenlp-latest.zip'
        extracted_files = download_and_extract(url=stanford_corenlp_latest_url, local_dir=str(stanford_dir))

        stanford_corenlp_jar_name = extracted_files[0].split('/')[0] + '.jar'
        stanford_corenlp_models_jar_name = extracted_files[0].split('/')[0] + '-models.jar'
        stanford_corenlp_jar_path = Path(stanford_dir) / extracted_files[0] / stanford_corenlp_jar_name
        stanford_corenlp_models_jar_path = Path(stanford_dir) / extracted_files[0] / stanford_corenlp_models_jar_name

        # Copy corenlp JAR file to lib directory.
        if not os.path.exists(stanford_corenlp_jar_path):
            os.makedirs(stanford_corenlp_jar_path)
        shutil.copy(stanford_corenlp_jar_path, stanford_dir)

        # Copy corenlp-models JAR file to lib directory.
        if not os.path.exists(stanford_corenlp_models_jar_path):
            os.makedirs(stanford_corenlp_models_jar_path)
        shutil.copy(stanford_corenlp_models_jar_path, stanford_dir)

        # Remove extracted Stanford CoreNLP directory.
        shutil.rmtree(Path(stanford_dir) / extracted_files[0].split('/')[0])

        log.info(
            f"Downloaded and extracted files: ['{stanford_corenlp_jar_path}', '{stanford_corenlp_models_jar_path}'].")

    # Download Stanford Shift-Reduce Constituency Parser.
    if input_params['download_srparser']:
        stanford_srparser_url = 'https://nlp.stanford.edu/software/stanford-srparser-2014-10-23-models.jar'
        if not os.path.exists(stanford_dir):
            os.makedirs(stanford_dir)
        download(url=stanford_srparser_url, local_dir=str(stanford_dir))
        log.info(
            f"Downloaded Stanford Shift-Reduce Constituency Parser library to '{stanford_dir}'.")

    # MongoDB Java Driver.
    if input_params['download_mongodb']:
        mongodb_url = 'https://repo1.maven.org/maven2/org/mongodb/mongo-java-driver/3.12.7/mongo-java-driver-3.12.7.jar'
        mongodb_dir = Path(ROOT_DIR) / 'lib' / 'mongodb'
        if not os.path.exists(mongodb_dir):
            os.makedirs(mongodb_dir)
        download(url=mongodb_url, local_dir=str(mongodb_dir))
        log.info(
            f"Downloaded MongoDB Java Driver to '{mongodb_dir}'.")

    # Word emebeddings.
    for model_name in ['glove', 'word2vec', 'fasttext']:
        if input_params[f"download_{model_name}"]:
            local_dir = Path(ROOT_DIR) / os.environ['DATA_PATH'] / 'word_embeddings' / model_name
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            vectors = download_gensim_vectors(model_name=model_name, local_dir=str(local_dir))
