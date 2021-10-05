#!/usr/bin/env python3

import argparse
import os
import time

from syn.helpers.argparser import all_operations_parser, sentence_model_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[all_operations_parser, sentence_model_parser],
        description='Build all datasets.'
    )

    args = parser.parse_args()

    return {
        'architecture': args.architecture,
        'task_list': args.task_list,
        'corpus_list': args.corpus_list
    }


def get_command(
        os_name: str = 'posix',
        arch: str = 'tree_lstm',
        task: str = 'prioritization',
        corpus: str = 'openOffice',
        balanced: bool = False
):
    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    arguments = f" -m syn.model.build.common.build_dataset --task {task} --corpus {corpus} --architecture {arch}"
    if balanced:
        arguments = f"{arguments} --balance_data"
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

    log.info(f"Building all datasets ...")

    architecture = input_params['architecture']
    task_list = input_params['task_list'].split(",") if '' != input_params['task_list'] else \
        os.environ["TASK_NAME"].split(",")
    corpus_list = input_params['corpus_list'].split(",") if '' != input_params['corpus_list'] else \
        os.environ["CORPUS_NAME"].split(",")

    for corpus in corpus_list:
        log.info(f"Building datasets for corpus: '{corpus}' ...")
        for task in task_list:
            tic = time.time()
            log.info(f"Building datasets for task: '{task}' ...")
            cmd1 = get_command(os.name, architecture, task, corpus, False)
            cmd2 = get_command(os.name, architecture, task, corpus, True)

            # Run command.
            log.info(f"Running command: '{cmd1}'")
            os.system(cmd1)
            log.info(f"Building unbalanced dataset total execution time = {((time.time() - tic) / 60)} minutes")
            tic = time.time()
            log.info(f"Running command: '{cmd2}'")
            os.system(cmd2)
            log.info(f"Building balanced dataset total execution time = {((time.time() - tic) / 60)} minutes")

    final_time = time.time()
    log.info(f"Building all datasets total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")


if __name__ == '__main__':
    main()
