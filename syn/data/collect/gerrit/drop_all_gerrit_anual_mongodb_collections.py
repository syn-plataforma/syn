#!/usr/bin/env python3
import os
import time

from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Establece los parámetros que se enviarán en la petición.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[os.environ['GERRIT_DB_NAME']]
    log.debug(f"Existent collections in '{db.name}': {str(db.list_collection_names())}")

    for project in os.environ["GERRIT_PROJECT_NAME"].split(","):
        for year in range(int(os.environ['GERRIT_FIRST_CREATION_YEAR']), int(os.environ['GERRIT_LAST_CREATION_YEAR'])):
            col = db[f"{project.lower()}_{year}_{year + 1}"]
            if col.name in db.list_collection_names():
                log.info(f"Dropping collection {col.name} ...")
                db.drop_collection(col.name)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
