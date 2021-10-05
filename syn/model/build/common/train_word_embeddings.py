"""Train word embeddings"""

import argparse
import os
import time
from pathlib import Path

from gensim.models import Word2Vec, FastText

from definitions import ROOT_DIR
from syn.helpers.argparser import dataset_parser, embeddings_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.nlp.vocabulay import load_tokens
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()

log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[dataset_parser, embeddings_parser],
        description='Train word embeddings.'
    )

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'dataset_name': args.dataset_name,
        'query_limit': args.query_limit,
        'embeddings_model': args.embeddings_model,
        'embeddings_size': args.embeddings_size
    }


def train_glove(sentences, size):
    raise NotImplementedError("This method is not implemented.")


def train_word2vec(sentences, size):
    return Word2Vec(sentences=sentences, size=size, window=5, min_count=1, iter=10, workers=4)


def train_fasttext(sentences, size):
    model = FastText(size=size, window=5, min_count=1, workers=4)
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=10)

    return model


def train_model(model_name: str = 'word2vec'):
    model_function = {
        'glove': train_glove,
        'word2vec': train_word2vec,
        'fasttext': train_fasttext
    }
    return model_function[model_name]


if __name__ == "__main__":
    log.info(f"Training word embeddings ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Loads dataset.
    log.info(f"Loading issues descriptions ...")
    description_tokens = load_tokens(
        database_name=input_params['corpus'],
        collection_name=input_params['dataset_name'],
        query_limit=input_params['query_limit']
    )
    log.info(f"Issues descriptions loaded.")

    # Build restartable iterable (not just a generator), to allow the algorithm to stream over your dataset multiple
    # times.
    corpus = []
    for sentence_tokens in description_tokens['tokens']:
        for sentence in sentence_tokens:
            corpus.append(sentence)

    log.info(f"Training {input_params['embeddings_model']} model ...")

    local_dir = Path(ROOT_DIR) / os.environ['DATA_PATH'] / 'word_embeddings' / input_params['embeddings_model']
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    base_filename = f"{input_params['embeddings_model']}-{input_params['corpus']}-{input_params['embeddings_size']}"
    trained_model = train_model(input_params['embeddings_model'])(corpus, input_params['embeddings_size'])
    # trained_model.save(f"{str(Path(local_dir) / base_filename)}.model")
    trained_model.wv.save_word2vec_format(f"{str(Path(local_dir) / base_filename)}.txt", binary=False)
    log.info(f"Word embeddings trained and saved: '{base_filename}'.")

    log.info(f"Training word embeddings total time = {((time.time() - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
