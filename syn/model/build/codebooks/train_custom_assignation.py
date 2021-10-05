import os
import time

from syn.helpers.hyperparams import get_codebooks_input_params
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.system import check_same_python_module_already_running
from syn.model.build.codebooks.codebooks_task import IssueAssignerTrain

log = set_logger()

if __name__ == "__main__":
    log.info(f"Training model ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_codebooks_input_params()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    train = load_dataframe_from_mongodb(
        database_name=os.environ.get('BUGZILLA_MONGODB_DATABASE_NAME', input_params['dataset']['corpus']),
        collection_name=input_params['dataset']['train']
    )

    opened = load_dataframe_from_mongodb(
        database_name=os.environ.get('BUGZILLA_MONGODB_DATABASE_NAME', input_params['dataset']['corpus']),
        collection_name=input_params['dataset']['opened']
    )

    trainer = IssueAssignerTrain(train, opened, input_params)
    result = trainer.run_and_save()

    log.info(f"Training model total time = {((time.time() - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
