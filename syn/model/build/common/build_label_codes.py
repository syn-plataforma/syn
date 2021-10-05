#!/usr/bin/env python3

import argparse
import os
import time

import pandas as pd
from pymongo import MongoClient

from syn.helpers.argparser import dataset_parser, similarity_task_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.task import get_label_column_name, get_label_collection_name

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[dataset_parser, similarity_task_parser],
        description='Build label codes.'
    )

    args = parser.parse_args()

    return {
        'task': args.task,
        'corpus': args.corpus,
        'near_issues': args.near_issues
    }


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Building label codes ...")

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Query the database.
    label_column_name = get_label_column_name(input_params['task'], input_params['corpus'])
    if '' == label_column_name:
        raise NotImplementedError('This module only works with prioritization, classification and assignation tasks.')
    log.info(f"Label column name for task '{input_params['task']}' and corpus '{input_params['corpus']}':"
             f" {label_column_name}")

    # Load normalized_clear collection.
    projection = {'_id': 0, 'label': f"${label_column_name}"}

    collection_name = get_label_collection_name(input_params['task'], input_params['corpus'])
    if 'similarity' == input_params['task']:
        if not input_params['near_issues']:
            collection_name = f"similar_{collection_name}"
        else:
            collection_name = f"near_{collection_name}"

    df_labels = load_dataframe_from_mongodb(
        database_name=input_params['corpus'],
        collection_name=collection_name,
        projection=projection
    )

    # Group_by label column.
    labels_value_counts = df_labels['label'].value_counts()
    log.info(f"Number of distinct label values: {labels_value_counts.shape[0]}")

    df_distinct_labels = pd.DataFrame(
        data=labels_value_counts.keys().to_list(),
        columns=['label']
    )

    # converting type of label column to 'category'
    df_distinct_labels['label'] = df_distinct_labels['label'].astype('category')

    # Assigning numerical values and storing in another column
    df_distinct_labels['label_code'] = df_distinct_labels['label'].cat.codes

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client() if os.environ['WORK_ENVIRONMENT'] == 'aws' \
        else MongoClient(host='localhost', port=27017)

    db = mongodb_client[input_params['corpus']]
    col = db[f"{input_params['task']}_task_labels"]

    log.debug(f"Existent collections in '{db.name}': {str(db.list_collection_names())}")

    if col.name in db.list_collection_names():
        db.drop_collection(col.name)

    log.info(f"Inserting documents ...")

    inserted_documents = col.insert_many(df_distinct_labels.to_dict("records"))

    log.info(f"Inserted documents: {len(inserted_documents.inserted_ids)}")

    final_time = time.time()
    log.info(f"Building label codes total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
