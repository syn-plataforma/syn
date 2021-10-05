"""Perform hyperparemeters search"""

import os
import time

from syn.helpers.hyperparams import get_input_params
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running
from syn.model.build.common.task import GridSearch

log = set_logger()

if __name__ == "__main__":
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Searching hyperparameters ...")

    # Load parameter space.
    input_param_space = get_input_params()
    assert input_param_space is not None, f"No param space provided."

    objective = input_param_space['hyper_search_objective'].split(',')
    tuner = GridSearch(objective=objective, parameter_space=input_param_space)
    tuner.search()
    models = tuner.get_best_models(num_models=1)
    log.info(f"Best model for objective '{objective}: {models}")
    tuner.results_summary()

    log.info(f"Searching hyperparameters total time = {((time.time() - initial_time) / 60)} minutes")
