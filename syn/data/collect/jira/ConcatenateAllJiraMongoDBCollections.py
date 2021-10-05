#!/usr/bin/env python3
import os
import time

import pandas as pd
from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.gerrit import get_input_params
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

    # Load parameters.
    input_params = get_input_params()

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[os.environ.get('GERRIT_DB_NAME', input_params['mongo_params'].db_name)]

    gerrit_collections = db.list_collection_names()
    log.debug(f"Existent collections in '{db.name}': {str(db.list_collection_names())}")

    for project in os.environ["JIRA_PROJECT_NAME"].split(","):
        df_list = []
        for year in range(int(os.environ['JIRA_FIRST_CREATION_YEAR']), int(os.environ['JIRA_LAST_CREATION_YEAR'])):
            col = db[
                f"{input_params['mongo_params'].collection_name}"
                f"_{year}_{year + 1}"
            ]
            if col.name in gerrit_collections:
                tic = time.time()
                log.info(f"Retrieving Jira issues for year '{year}' ...")
                data = col.find()
                df = pd.DataFrame(list(data))
                df = df.drop(['_id'], axis=1)
                df_list.append(df)
                log.info(
                    f"Retrieving Jira issues for year '{year}' execution time = {((time.time() - tic) / 60)} minutes")

        df_concatenated = pd.concat(df_list)
        table_dict = df_concatenated.to_dict("records")
        db[f"{project.lower()}_all"].insert_many(table_dict)

    final_time = time.time()
    log.info(f"Retrieving Gerrit issues total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
