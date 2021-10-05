#!/usr/bin/env python3
import argparse
import os
import sys
import time

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running


def get_input_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_year", default=1998, type=int)
    parser.add_argument("--end_year", default=2021, type=int)
    parser.add_argument("--corpus", default='', type=str)
    parser.add_argument('--get-trees', default=True, dest='get-trees', action='store_true',
                        help="Obtener árboles sintácticos.")
    parser.add_argument('--no-get-trees', dest='get-trees', action='store_false',
                        help="No obtener árboles sintácticos.")
    parser.add_argument('--get-embeddings', default=True, dest='get-embeddings', action='store_true',
                        help="Obtener embeddings de las hojas de los árboles sintácticos.")
    parser.add_argument('--no-get-embeddings', dest='get-embeddings', action='store_false',
                        help="No obtener embeddings de las hojas de los árboles sintácticos.")
    parser.add_argument('--get-coherence', default=True, dest='get-coherence', action='store_true',
                        help="Obtener scores de los árboles sintácticos.")
    parser.add_argument('--no-get-coherence', dest='get-coherence', action='store_false',
                        help="No obtener scores de los árboles sintácticos.")

    args = parser.parse_args()

    return {
        'start_year': args.start_year,
        'end_year': args.end_year,
        'corpus': args.corpus,
        'get-trees': args.__getattribute__('get-trees'),
        'get-embeddings': args.__getattribute__('get-embeddings'),
        'get-coherence': args.__getattribute__('get-coherence')
    }


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Logger.
    log = set_logger()

    log.debug(f"\n[START OF EXECUTION]")

    load_environment_variables()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Years range.
    input_params = get_input_params()

    # Databases.
    databases = [input_params['corpus']] if input_params['corpus'] != '' \
        else os.environ["EMBEDDING_MONGODB_DATABASE_NAME"].split(",")

    # Java class.
    java_class_name = "UpdateMongoDBNLPFields"

    # Control params.
    model_param = f"--pm {'corenlp'}" if (input_params['get-coherence'] and input_params['get-trees']) \
        else f"--pm {'srparser'}"
    trees_param = "--get-trees" if input_params['get-trees'] else "--no-get-trees"
    embeddings_param = "--get-embeddings " if (input_params['get-embeddings'] and input_params['get-trees']) \
        else "--no-get-embeddings"
    coherence_param = "--get-coherence" if (input_params['get-coherence'] and input_params['get-trees']) \
        else "--no-get-coherence"

    # Defines Python executable.
    python_exe = os.environ.get('PYTHON_EXECUTABLE', sys.executable)

    # Loop for obtain tokens number.
    tokens_initial_time = time.time()
    log.info(f"Updating NLP fields ...")
    for db in databases:
        log.info(f"\nProcessing database: '{db}'.")
        for year in range(input_params['start_year'], input_params['end_year']):
            log.info(f"\n[FOR LOOP] Processing years: {year} - {year + 1}")
            cmd = f"{python_exe} UpdateVectorizedMongoDBCollection.py --jcn {java_class_name}" \
                  f" --mh {os.environ['MONGO_HOST_IP']}" \
                  f" --mp {os.environ['MONGO_PORT']}" \
                  f" --db {db}" \
                  f" --c {os.environ['EMBEDDING_MONGODB_COLLECTION_NAME']}" \
                  f" --cl {os.environ['EMBEDDING_MONGODB_COLUMN_NAME']}" \
                  f" --sy {year}" \
                  f" --ey {year + 1} " \
                  f"--mnt {os.environ['EMBEDDING_MONGODB_MAX_NUM_TOKENS']} " \
                  f"{model_param} {trees_param} {embeddings_param} {coherence_param}"

            # Run command.
            log.info(f"Running command: '{cmd}'.")
            os.system(cmd)
    log.info(f"Updating NLP fields total execution time = {((time.time() - tokens_initial_time) / 60)} minutes")

    log.debug(f"\n[END OF EXECUTION]")
    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
