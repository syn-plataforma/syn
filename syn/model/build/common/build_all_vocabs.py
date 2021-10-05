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
        'corpus_list': args.corpus_list
    }


def get_command(
        os_name: str = 'posix',
        corpus: str = 'openOffice'
):
    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    arguments = f" -m syn.model.build.common.build_vocab --corpus {corpus}"
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

    log.info(f"Building all vocabs ...")

    corpus_list = input_params['corpus_list'] if '' != input_params['corpus_list'] else \
        os.environ["CORPUS_NAME"].split(",")

    for corpus in corpus_list:
        tic = time.time()
        log.info(f"Building vocab for corpus: '{corpus}' ...")
        cmd = get_command(os.name, corpus)

        # Run command.
        log.info(f"Running command: '{cmd}'")
        os.system(cmd)
        log.info(f"Building vocab total execution time = {((time.time() - tic) / 60)} minutes")

    final_time = time.time()
    log.info(f"Building all vocabs total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")


if __name__ == '__main__':
    main()
