"""Synthesize  hyperparemeters search results"""

import os
import time

from syn.helpers.hyperparams import get_input_params
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running
from syn.model.build.common.task import AggregatedMetrics

log = set_logger()

if __name__ == "__main__":
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Synthesizing results ...")

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Aggregate metrics from MongoDB
    aggregated_metrics = AggregatedMetrics(
        dbname=input_params['source_params']['database_name'],
        collection=input_params['source_params']['collection_name'],
        task=input_params['dataset']['task'],
        corpus=input_params['dataset']['corpus']
    )

    aggregated_metrics.results_summary(
        metric_name=['accuracy', 'precision_micro', 'recall_micro', 'f1_micro'],
        sort_by=['accuracy'],
        num_models=0
    )
    log.info(f"MODULE EXECUTED.")
