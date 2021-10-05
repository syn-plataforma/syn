"""Perform hyperparemeters search"""

import argparse
import os
import time

from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running
from syn.model.build.common.task import AggregatedMetrics

log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Hyperparameters search summary.')

    parser.add_argument('--db_name', default='tasks', type=str, help='Database name.')
    parser.add_argument('--collection_name', default='experiments', type=str, help='Collection name.')
    parser.add_argument('--task', default='prioritization', type=str, help='Task name.')
    parser.add_argument('--corpus', default='openOffice', type=str, help='Corpus.')
    parser.add_argument('--metrics', default='accuracy,precision_micro,recall_micro,f1_micro', type=str,
                        help='Retrieved metrics.')
    parser.add_argument('--objective', default='accuracy', type=str, help='Objective function.')
    parser.add_argument('--num_models', default=10, type=int, help='Number of retrieved models.')
    parser.add_argument('--fields', default='embeddings_model,embeddings_size,embeddings_pretrained,'
                                            'codebooks_n_codewords,use_structured_data,classifier',
                        type=str, help='Retrieved fields.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'task': args.task,
        'corpus': args.corpus,
        'metrics': args.metrics,
        'objective': args.objective,
        'num_models': args.num_models,
        'fields': args.fields
    }


if __name__ == "__main__":
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Generating hyperparameters search summary ...")

    # Load parameter space.
    input_param_space = get_input_params()
    assert input_param_space is not None, f"No param space provided."

    metrics = AggregatedMetrics(
        dbname=input_param_space['db_name'],
        collection=input_param_space['collection_name'],
        task=input_param_space['task'],
        corpus=input_param_space['corpus']
    )

    metrics.results_summary(
        metric_name=input_param_space['metrics'].split(','),
        sort_by=input_param_space['objective'].split(','),
        num_models=input_param_space['num_models'],
        fields=input_param_space['fields'].split(',') if input_param_space['fields'] != '' else None
    )

    log.info(f"Generating hyperparameters search summary total time = {((time.time() - initial_time) / 60)} minutes")
