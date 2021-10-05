#!/usr/bin/env python3

import argparse
import os
import time

from syn.helpers.argparser import all_operations_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[all_operations_parser],
        description='Build all vocabularies.'
    )

    args = parser.parse_args()

    return {
        'corpus_list': args.corpus_list,
        'embeddings_model_list': args.embeddings_model_list,
        'embeddings_size_list': args.embeddings_size_list
    }


def get_command(
        os_name: str = 'posix',
        corpus: str = 'openOffice',
        model: str = 'word2vec',
        size: int = 100
):
    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    arguments = f" -m syn.model.build.common.train_word_embeddings --corpus {corpus} " \
                f"--embeddings_model {model} --embeddings_size {size}"
    cmd = {
        'posix': f"python3 {arguments}",
        'nt': f"python {arguments}",
        'java': ''
    }

    return cmd[os_name]


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_input_params()

    log.info(f"Training all word embeddings ...")

    corpus_list = input_params['corpus_list'].split(",") if '' != input_params['corpus_list'] else \
        os.environ["CORPUS_NAME"].split(",")
    embeddings_model_list = input_params['embeddings_model_list'].split(",") if '' != input_params[
        'embeddings_model_list'] else \
        os.environ["EMBEDDINGS_MODEL"].split(",")
    embeddings_size_list = input_params['embeddings_size_list'].split(",") if '' != input_params[
        'embeddings_size_list'] else \
        os.environ["EMBEDDINGS_SIZE"].split(",")

    for corpus in corpus_list:
        log.info(f"Training word embeddings for corpus: '{corpus}' ...")
        for model in embeddings_model_list:
            log.info(f"Training word embeddings for model: '{model}' ...")
            for size in embeddings_size_list:
                tic = time.time()
                log.info(f"Training word embeddings of size: '{int(size)}' ...")
                cmd = get_command(os.name, corpus, model, int(size))

                # Run command.
                log.info(f"Running command: '{cmd}'")
                os.system(cmd)
                log.info(f"Training word embeddings total execution time = {((time.time() - tic) / 60)} minutes")

    final_time = time.time()
    log.info(f"Training all word embeddings total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")


if __name__ == '__main__':
    main()
