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
    parser = argparse.ArgumentParser(
        description='Concatenate MongodDB collections in a single collection.')

    parser.add_argument('--db_name', default='bugzilla', type=str, help='Database name.')
    parser.add_argument('--collection_name', default='eclipse', type=str, help='Collection name.')

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

    log.info("Concatenating collections ...")

    # Load parameters.
    input_params = get_input_params()

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[os.environ.get(input_params['db_name'])]

    collections = db.list_collection_names()
    log.debug(f"Existent collections in '{db.name}': {str(db.list_collection_names())}")

    project_name = input_params['collection_name']
    output_collection = f"{project_name}_all"

    # initialize dataframe list
    df_list = []
    for year in range(int(os.environ[f"{project_name.upper()}_FIRST_CREATION_YEAR"]),
                      int(os.environ[f"{project_name.upper()}_LAST_CREATION_YEAR"])):
        current_collection_name = f"{project_name}_{year}_{year + 1}"
        col = db[current_collection_name]
        if col.name in collections:
            tic = time.time()
            log.info(f"Retrieving collection '{current_collection_name}' ...")
            data = col.find({}, {'_id': 0})
            df = pd.DataFrame(list(data))
            log.info(f"Number of documents in collecction '{current_collection_name}': {df.shape[0]}")
            df_list.append(df)
            log.info(f"Retrieving collection '{current_collection_name}' "
                     f"execution time = {((time.time() - tic) / 60)} minutes")

        df_concatenated = pd.concat(df_list)
        table_dict = df_concatenated.to_dict("records")
        if output_collection in collections:
            db.drop_collection(output_collection)
        db[output_collection].insert_many(table_dict)
    final_time = time.time()
    log.info(f"Concatenating collections total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
