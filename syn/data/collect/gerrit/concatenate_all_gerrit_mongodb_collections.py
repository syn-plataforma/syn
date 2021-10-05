#!/usr/bin/env python3
import argparse
import os
import time

import pandas as pd
from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Retrieve clear and normalized clear collections.')

    parser.add_argument('--db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--collection_name', default='eclipse', type=str, help='Gerrit collection name.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name
    }


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[os.environ.get('GERRIT_DB_NAME', input_params['db_name'])]

    gerrit_collections = db.list_collection_names()
    log.debug(f"Existent collections in '{db.name}': {str(db.list_collection_names())}")

    for project in os.environ["GERRIT_PROJECT_NAME"].split(","):
        df_list = []
        for year in range(int(os.environ['GERRIT_FIRST_CREATION_YEAR']), int(os.environ['GERRIT_LAST_CREATION_YEAR'])):
            col = db[
                f"{input_params['collection_name']}"
                f"_{year}_{year + 1}"
            ]
            if col.name in gerrit_collections:
                tic = time.time()
                log.info(f"Retrieving Gerrit issues for year '{year}' ...")
                data = col.find({}, {'_id': 0})
                df = pd.DataFrame(list(data))
                log.info(f"Gerrit issues for year '{year}': {df.shape[0]}")
                df_list.append(df)
                log.info(
                    f"Retrieving Gerrit issues for year '{year}' execution time = {((time.time() - tic) / 60)} minutes")

        df_concatenated = pd.concat(df_list)
        table_dict = df_concatenated.to_dict("records")
        if project.lower() in gerrit_collections:
            db.drop_collection(project.lower())
        db[project.lower()].insert_many(table_dict)
    final_time = time.time()
    log.info(f"Retrieving Gerrit issues total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
