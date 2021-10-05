"""
Preprocessing script for SYN data.
"""

import os
import time
from pathlib import Path

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.nlp.parsers import get_input_params
from syn.helpers.system import check_same_python_module_already_running, get_java_classpath


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Logger.
    log = set_logger()
    log.debug(f"\n[START OF EXECUTION]")

    load_environment_variables()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Input parameters.
    input_params = get_input_params()

    # Defines javac executable.
    java_exe = Path(os.environ['JAVA_HOME']) / 'bin' / 'java.exe'

    # Command.
    cmd = f"{java_exe} -cp {get_java_classpath()} {input_params['nlp_params'].java_class_name} " \
          f"-host {input_params['mongo_params'].host} " \
          f"-port {input_params['mongo_params'].port} " \
          f"-dbName {input_params['mongo_params'].db_name} " \
          f"-collName {input_params['mongo_params'].collection_name} " \
          f"-startYear {input_params['filter_params'].start_year} " \
          f"-endYear {input_params['filter_params'].end_year} " \
          f"-textColumnName {input_params['filter_params'].column_name} " \
          f"-maxNumTokens {input_params['nlp_params'].max_num_tokens} " \
          f"-parserModel {input_params['nlp_params'].parser_model} " \
          f"-createTrees {input_params['nlp_params'].get_trees} " \
          f"-calcEmbeddings {input_params['nlp_params'].get_embeddings} " \
          f"-calcCoherence {input_params['nlp_params'].get_coherence}"

    log.info(f"Running command: '{cmd}'")

    # Run command.
    os.system(cmd)

    log.info(f"\n[END OF EXECUTION]")
    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
