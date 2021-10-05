"""Read, split and save the dataset for SYN model"""

import argparse
import os
from pathlib import Path

from syn.helpers.argparser import common_parser, vocabulary_parser, embeddings_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.nlp.embeddings import filter_word_embeddings, get_word_embeddings_filename, \
    get_filtered_word_embeddings_filename
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.nlp.vocabulay import load_vocabulary

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[common_parser, vocabulary_parser, embeddings_parser],
        description='Filter word embeddings.'
    )

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'query_limit': args.query_limit,
        'vocabulary_name': args.vocabulary_name,
        'embeddings_model': args.embeddings_model,
        'embeddings_size': args.embeddings_size
    }


if __name__ == "__main__":
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Only have pre-trained word embeddings of dimension 300 for Word2Vec and FastText.
    if 300 != input_params['embeddings_size'] \
            and ('word2vec' == input_params['embeddings_model'] or 'fasttext' == input_params['embeddings_model']):
        raise NotImplementedError(
            f"This functionality is not implemented for {input_params['embeddings_model']} model and "
            f"pre-trained word embeddings of dimension {input_params['embeddings_size']}. Try "
            f"trained word embeddings or change dimension to 300.")

    # Load vocabulary.
    log.info(f"Loading vocabulary ...")
    vocab = load_vocabulary(
        database_name=input_params['corpus'],
        collection_name=input_params['vocabulary_name'],
        query_limit=input_params['query_limit']
    )
    log.info(f"Vocabulary loaded.")

    # Filter word embeddings.
    we_filename = get_word_embeddings_filename(
        model=input_params['embeddings_model'],
        size=input_params['embeddings_size']
    )
    filtered_we_filename = get_filtered_word_embeddings_filename(
        corpus=input_params['corpus'],
        model=input_params['embeddings_model'],
        size=input_params['embeddings_size']
    )

    we_dir = Path(os.environ.get('DATA_PATH')) / 'word_embeddings'
    we_origin_path = Path(we_dir) / input_params['embeddings_model'] / we_filename
    we_filtered_path = Path(we_dir) / input_params['embeddings_model'] / filtered_we_filename

    word_embeddings_filtered = filter_word_embeddings(
        source=we_origin_path,
        dest=we_filtered_path,
        vocab=vocab
    )

    assert os.path.exists(we_filtered_path)
    log.info(f"MODULE EXECUTED.")
