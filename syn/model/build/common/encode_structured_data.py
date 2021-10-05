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
    parser.add_argument('--collection_name', default='normalized_clear', type=str, help='Collection name.')
    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'collection_name': args.collection_name
    }


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Encoding structured data ...")

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Query the database.
    structured_data_column_name = os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')

    if len(structured_data_column_name) == 0:
        raise ValueError('No structured data column names defined.')
    log.info(f"Structured data column name: {structured_data_column_name}")

    # Load normalized_clear collection.
    projection = {'_id': 0}

    for column in structured_data_column_name:
        projection[column] = 1

    log.info(f"projection:{projection}")

    df_structured_data = load_dataframe_from_mongodb(
        database_name=input_params['corpus'],
        collection_name=input_params['collection_name'],
        projection=projection
    )

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params['corpus']]

    # Group_by structured data column.
    for column in structured_data_column_name:
        df = pd.DataFrame(columns=[column])
        column_value_counts = df_structured_data[column].value_counts()

        log.info(f"Number of distinct values in column '{column}': {column_value_counts.shape[0]}")

        df[column] = column_value_counts.keys().to_list()

        # Assigning numerical values and storing in another column
        df[f"{column}_code"] = df[column].index

        col = db[f"{column}_codes"]

        if col.name in db.list_collection_names():
            db.drop_collection(col.name)

        log.info(f"Inserting documents ...")

        inserted_documents = col.insert_many(df.to_dict("records"))

        log.info(f"Inserted documents: {len(inserted_documents.inserted_ids)}")

    final_time = time.time()
    log.info(f"Encoding structured data total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
