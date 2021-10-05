"""Build vocabulary for SYN corpus"""

import argparse
import os

from tqdm import tqdm

from syn.helpers.argparser import common_parser, vocabulary_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.nlp.vocabulay import load_tokens, get_vocabulary_from_mongodb, save_vocabulary
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[common_parser, vocabulary_parser],
        description='Filter word embeddings.'
    )

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'query_limit': args.query_limit,
        'vocabulary_name': args.vocabulary_name,
        'tokens_collection': args.tokens_collection,
    }


if __name__ == "__main__":
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_input_params()

    # Loads dataset.
    log.info(f"Loading Dataframe ...")
    df = load_tokens(
        database_name=input_params['corpus'],
        collection_name=input_params['tokens_collection'],
        query_limit=input_params['query_limit']
    )
    log.info(f"Dataframe columns: {df.columns}")
    log.info(f"Dataframe loaded.")

    # Build vocabulary.
    log.info(f"Building vocabulary ...")
    vocab = set()
    for row in tqdm(df['tokens'], total=len(df['tokens']), desc='rows'):
        tmp_set = get_vocabulary_from_mongodb(row)
        vocab.update(tmp_set)

    log.info(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary.
    inserted_documents = save_vocabulary(
        database_name=input_params['corpus'],
        collection_name=input_params['vocabulary_name'],
        vocabulary=vocab
    )
    assert inserted_documents == len(vocab)
    log.info(f"MODULE EXECUTED.")
