#!/usr/bin/env python3

import argparse
import os
import time

import pandas as pd
from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Encode structured data.')

    parser.add_argument('--corpus', default='bugzilla', type=str, help='Database name.')
    parser.add_argument('--collection_list', default='pairs,similar_pairs,near_pairs', type=str, help='Collection name.')
    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'collection_list': args.collection_list
    }


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Encoding label ...")

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Query the database.
    column_name = 'dec'

    # Load normalized_clear collection.
    projection = {'_id': 0, 'dec': 1}
    log.info(f"projection:{projection}")

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params['corpus']]

    for collection_name in input_params['collection_list'].split(','):
        if collection_name not in db.list_collection_names():
            log.info(f"Collection '{collection_name}' not exists in database '{db.name}'.")
            continue
        df_labels = load_dataframe_from_mongodb(
            database_name=input_params['corpus'],
            collection_name=collection_name,
            projection=projection
        )

        # Group_by label column.
        labels_value_counts = df_labels['dec'].value_counts()
        log.info(f"Number of distinct label values: {labels_value_counts.shape[0]}")

        df_distinct_labels = pd.DataFrame(
            data=labels_value_counts.keys().to_list(),
            columns=['dec']
        )

        # converting type of label column to 'category'
        df_distinct_labels['dec'] = df_distinct_labels['dec'].astype('category')

        # Assigning numerical values and storing in another column
        df_distinct_labels['dec_code'] = df_distinct_labels['dec'].cat.codes

        col = db[f"{collection_name}_dec_codes"]

        if col.name in db.list_collection_names():
            db.drop_collection(col.name)

        log.info(f"Inserting documents ...")

        inserted_documents = col.insert_many(df_distinct_labels.to_dict("records"))

        log.info(f"Inserted documents: {len(inserted_documents.inserted_ids)}")

    final_time = time.time()
    log.info(f"Encoding label total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
