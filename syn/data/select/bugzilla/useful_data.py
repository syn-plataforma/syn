import argparse
import datetime
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
    parser = argparse.ArgumentParser(description='Search useful issues.')

    parser.add_argument('--db_name', default='bugzilla', type=str, help='Bugzilla database name.')
    parser.add_argument('--collection_name', default='eclipse_all', type=str, help='Bugzilla collection name.')
    parser.add_argument('--output_collection_name', default='clear', type=str,
                        help='Bugzilla output collection name.')
    parser.add_argument('--batch_size', default=10000, type=int, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_collection_name': args.output_collection_name,
        'batch_size': args.batch_size
    }


def get_python_datetime(string_date):
    return datetime.datetime.strptime(string_date, "%Y-%m-%dT%H:%M:%SZ")


def main():
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Incializa las variables que almacenarán los argumentos de entrada.
    input_params = get_input_params()

    # Establece los parámetros que se enviarán en la petición.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params['db_name']]
    source_collection = db[input_params['collection_name']]

    # retrieve only closed issues
    aggregation_match = {"$match": {
        '$and': [
            {
                '$or': [
                    {'status': {'$eq': 'RESOLVED'}},
                    {'status': {'$eq': 'VERIFIED'}},
                    {'status': {'$eq': 'CLOSED'}}
                ]
            },
            {'comments': {'$exists': 'true'}},
            {'comments': {'$ne': 'null'}},
            {'comments': {'$ne': ""}},
            {'comments': {'$not': {'$size': 0}}},
            {'comments.count': {'$eq': 0}}
        ]
    }}
    aggregation_unwind = {"$unwind": "$comments"}

    log.info(f"Aggregation match: {aggregation_match}")

    projection = {
        "creation_time": 1,
        "short_desc": "$summary",
        "bug_status": "$status",
        "bug_id": "$id",
        "dup_id": "$dupe_of",
        "product": 1,
        "priority": 1,
        "component": 1,
        "last_change_time": 1,
        "bug_severity": "$severity",
        "description": "$comments.text",
        "assigned_to": 1
    }

    log.info(f"Aggregation projection: {projection}")

    pipeline = [
        aggregation_unwind,
        aggregation_match,
        {'$project': projection}
    ]

    mongo_cursor = source_collection.aggregate(pipeline)

    df = pd.DataFrame(list(mongo_cursor))

    # remove descriptions with only white spaces
    df_clean = df.applymap(lambda x: x.strip() if isinstance(x, str) else x).copy()
    df_final = df_clean[df_clean['description'].str.len() > 0].copy()

    # transform string dates to Python dates
    df_final['creation_ts'] = df_final['creation_time'].apply(lambda x: get_python_datetime(x))
    df_final['delta_ts'] = df_final['last_change_time'].apply(lambda x: get_python_datetime(x))
    df_final.drop(['creation_time', 'last_change_time'], axis=1, inplace=True)

    # Drop collection if already exists.
    if input_params['output_collection_name'] in db.list_collection_names():
        log.info(f"Dropping collection {input_params['output_collection_name']} ...")
        db.drop_collection(input_params['output_collection_name'])

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df_final.shape[0] / input_params['batch_size'])
    batches = np.array_split(df_final.loc[df_final['bug_id'] != -1], num_batches)

    # Insert documents with bug_id in MongoDB.
    inserted_docs_number = 0
    for batch in batches:
        log.info(f"Inserting documents ...")
        inserted_documents = db[input_params['output_collection_name']].insert_many(batch.to_dict("records"))
        inserted_docs_number += len(inserted_documents.inserted_ids)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
