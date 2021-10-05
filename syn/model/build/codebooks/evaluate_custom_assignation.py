import os
import time

from syn.helpers.environment import load_environment_variables
from syn.helpers.hyperparams import get_input_params
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.system import check_same_python_module_already_running
from syn.model.build.codebooks.codebooks_task import IssueAssignerEvaluate

load_environment_variables()
log = set_logger()

if __name__ == "__main__":
    log.info(f"Evaluating model ...")
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    test = load_dataframe_from_mongodb(
        database_name=os.environ.get('BUGZILLA_MONGODB_DATABASE_NAME', input_params['dataset']['corpus']),
        collection_name=input_params['dataset']['test']
    )

    evaluator = IssueAssignerEvaluate(test, None, None, input_params)
    result = evaluator.run_and_save()

    log.info(f"Evaluating model total time = {((time.time() - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
