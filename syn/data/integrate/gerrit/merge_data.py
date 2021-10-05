#!/usr/bin/env python3
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Merge Gerrit and Bugzilla issues.')

    parser.add_argument('--gerrit_db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--gerrit_collection_name', default='eclipse_useful', type=str, help='Gerrit collection name.')
    parser.add_argument('--output_collection_name', default='eclipse_merged', type=str,
                        help='Gerrit output collection name.')
    parser.add_argument('--bugzilla_db_name', default='bugzilla', type=str, help='Bugzilla database name.')
    parser.add_argument('--bugzilla_collection_name', default='normalized_clear', type=str,
                        help='Bugzilla collection name.')
    parser.add_argument('--batch_size', default=10000, type=str, help='Batch size.')

    args = parser.parse_args()

    return {
        'gerrit_db_name': args.gerrit_db_name,
        'gerrit_collection_name': args.gerrit_collection_name,
        'bugzilla_db_name': args.bugzilla_db_name,
        'bugzilla_collection_name': args.bugzilla_collection_name,
        'output_collection_name': args.output_collection_name,
        'batch_size': args.batch_size
    }


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    log.info(f"Merging Gerrit and Bugzilla issues ...")

    # MongoDB client.
    mongodb_client: MongoClient = get_default_mongo_client()

    # Load Gerrit data.
    log.info(f"Loading Gerrit issues ...")
    tic = time.time()
    gerrit_db = mongodb_client[os.environ.get('GERRIT_DB_NAME', input_params['gerrit_db_name'])]
    gerrit_collection = gerrit_db[input_params['gerrit_collection_name']]
    gerrit_data = gerrit_collection.find({}, {'_id': 0})
    df_gerrit = pd.DataFrame(list(gerrit_data))
    log.info(f"Loading Gerrit issues total time: {(time.time() - tic) / 60} minutes.")

    # Check empty Dataframe.
    if 0 == df_gerrit.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{gerrit_db.name}.{gerrit_collection.name}' collection.")

    # Load Bugzilla data.
    log.info(f"Loading Bugzilla issues ...")
    tic = time.time()
    bugzilla_db = mongodb_client[os.environ.get('BUGZILLA_MONGODB_DATABASE_NAME', input_params['bugzilla_db_name'])]
    bugzilla_collection = bugzilla_db[input_params['bugzilla_collection_name']]
    bugzilla_data = bugzilla_collection.find({}, {'_id': 0, 'bug_id': 1})
    df_bugzilla = pd.DataFrame(list(bugzilla_data))
    log.info(f"Loading Gerrit issues total time: {(time.time() - tic) / 60} minutes.")

    # Check empty Dataframe.
    if 0 == df_bugzilla.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{bugzilla_db.name}.{bugzilla_collection.name}' collection.")

    # Join on column 'bug_id'.
    log.info(f"Joining Gerrit and Bugzilla Dataframes ...")
    tic = time.time()
    df_joined = df_gerrit.merge(df_bugzilla, left_on='bug_id', right_on='bug_id')
    log.info(f"Joining Gerrit and Bugzilla Dataframes total time: {(time.time() - tic) / 60} minutes.")

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df_joined.shape[0] / input_params['batch_size'])
    batches = np.array_split(df_joined, num_batches)

    output_collection = gerrit_db[input_params['output_collection_name']]

    # Drop collection if already exists.
    if input_params['output_collection_name'] in gerrit_db.list_collection_names():
        log.info(f"Dropping collection {input_params['output_collection_name']} ...")
        gerrit_db.drop_collection(input_params['output_collection_name'])

    inserted_docs_number = 0
    tic = time.time()
    for batch in batches:
        log.info(f"Inserting documents ...")
        inserted_documents = output_collection.insert_many(batch.to_dict("records"))
        inserted_docs_number += len(inserted_documents.inserted_ids)

    log.info(f"Inserted documents: {inserted_docs_number}")
    log.info(f"Inserting documents total time: {(time.time() - tic) / 60} minutes.")

    final_time = time.time()
    log.info(f"Merging Gerrit and Bugzilla issues total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
