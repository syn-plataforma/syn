#!/usr/bin/env python3

import argparse
import datetime
import math
import os
import time

import numpy as np
from pymongo import MongoClient

from syn.helpers.argparser import common_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[common_parser],
        description='Retrieve clear and normalized clear collections.'
    )

    parser.add_argument('--year', default=2000, type=int, help='Reference year.')
    parser.add_argument('--closed_states', default=True, dest='closed_states', action='store_true',
                        help="Filter by state CLOSED, RESOLVED, VERIFIED.")
    parser.add_argument('--no_closed_states', dest='closed_states', action='store_false', help="No filter by state.")

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'year': args.year,
        'closed_states': args.closed_states,
        'mongo_batch_size': args.mongo_batch_size
    }


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    log.info(f"Updating normalized_clear for year '{input_params['year']}' ...")

    # Load clear collection to get "assigned_to" field.
    query = {
        "creation_ts": {
            "$gte": datetime.datetime(input_params['year'], 1, 1).strftime('%Y-%m-%d'),
            "$lt": datetime.datetime(input_params['year'] + 1, 1, 1).strftime('%Y-%m-%d')
        }
    }

    if input_params['closed_states']:
        query['bug_status'] = {"$in": ["CLOSED", "RESOLVED", "VERIFIED"]}

    projection = {'_id': 0, 'bug_id': 1, 'assigned_to': 1}
    df_clear = load_dataframe_from_mongodb(
        database_name=input_params['corpus'],
        collection_name='clear',
        query=query,
        projection=projection
    )

    # Check empty Dataframe.
    if 0 == df_clear.shape[0]:
        raise ValueError(f"No documents have been retrieved from '{input_params['corpus']}.clear' collection for the "
                         f"year {input_params['year']}")

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params['corpus']]
    col_name = 'normalized_clear_updated' if input_params['closed_states'] else 'normalized_clear_all_states'
    col = db[col_name]

    # Load normalized_clear collection to get "assigned_to" field.
    df_normalized_clear = load_dataframe_from_mongodb(
        database_name=input_params['corpus'],
        collection_name='normalized_clear',
        query=query
    )

    # Check empty Dataframe.
    if 0 == df_clear.shape[0]:
        raise ValueError(f"No documents have been retrieved from '{input_params['corpus']}.normalized_clear' "
                         f"collection for the year {input_params['year']}")

    # Join on column 'bug_id'.
    updated_normalized_clear = df_normalized_clear.merge(df_clear, left_on='bug_id', right_on='bug_id')

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(updated_normalized_clear.shape[0] / input_params['batch_size'])
    batches = np.array_split(updated_normalized_clear, num_batches)

    inserted_docs_number = 0
    for batch in batches:
        log.info(f"Inserting documents ...")
        inserted_documents = col.insert_many(batch.to_dict("records"))
        inserted_docs_number += len(inserted_documents.inserted_ids)

    log.info(f"Inserted documents: {inserted_docs_number}")

    final_time = time.time()
    log.info(f"Updating normalized_clear year total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
