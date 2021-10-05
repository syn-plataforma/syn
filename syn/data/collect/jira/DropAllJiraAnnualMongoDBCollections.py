#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.mongodb import get_default_mongo_client


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()
    # Define el logger que se utilizará.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Establece los parámetros que se enviarán en la petición.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[os.environ['JIRA_MONGODB_DATABASE_NAME']]
    log.debug(f"Colecciones exitentes en '{db.name}': {str(db.list_collection_names())}")

    for project in os.environ["JIRA_PROJECT_NAME"].split(","):
        for year in range(2001, 2021):
            col = db[f"{project.lower()}_{year}_{year + 1}"]
            if col.name in db.list_collection_names():
                db.drop_collection(col.name)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
