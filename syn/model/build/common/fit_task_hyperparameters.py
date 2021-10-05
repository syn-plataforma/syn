"""Perform hyperparemeters fit"""

import os
import time

from syn.helpers.hyperparams import get_input_params
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running
from syn.model.build.common.task import ConsensusFit

log = set_logger()

if __name__ == "__main__":
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Fitting hyperparameters ...")

    # Load parameter space.
    input_param_space = get_input_params()
    assert input_param_space is not None, f"No param space provided."

    fitter = ConsensusFit(
        database_name='tasks',
        collection_name='experiments',
        corpus='openOffice',
        tasks_objectives={'duplicity': 'accuracy', 'prioritization': 'jaccard_micro'},
        common_hyperparams=[
            ['scheduler', 'learning_rate_param'],
            ['model', 'embeddings_model'],
            ['model', 'embeddings_size']
        ]
    )

    best_common_hyperparameters = fitter.get_best_common_hyperparameters()
    log.info(f"Best common hyperparameters: {best_common_hyperparameters}")
    best_specific_hyperparameters = fitter.get_best_specific_hyperparameters_rank()
    log.info(f"Best specific hyperparameters: {best_specific_hyperparameters}")

    log.info(f"Best common hyperparameters summary:")
    fitter.common_hyperparameters_rank_summary()
    log.info(f"Task losses summary:")
    fitter.task_losses_summary()
    log.info(f"Best specific hyperparameters summary:")
    fitter.specific_hyperparameters_rank_summary()

    log.info(f"Fitting hyperparameters total time = {((time.time() - initial_time) / 60)} minutes")
