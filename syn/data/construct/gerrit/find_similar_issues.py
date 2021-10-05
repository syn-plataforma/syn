#!/usr/bin/env python3
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Find similar issues.')

    parser.add_argument('--db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--collection_name', default='eclipse_merged', type=str, help='Gerrit collection name.')
    parser.add_argument('--output_collection_name', default='eclipse_similarities', type=str,
                        help='Gerrit similarity collection name.')
    parser.add_argument('--similarity_threshold', default=0.5, type=float, help='Similarity threshold.')
    parser.add_argument('--batch_size', default=10000, type=str, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_collection_name': args.output_collection_name,
        'similarity_threshold': args.similarity_threshold,
        'batch_size': args.batch_size
    }


def mutual_score(file_list_1: list = None, file_list_2: list = None) -> float:
    mutual = 0.0
    intersection = list(set(file_list_1).intersection(file_list_2))
    numerator = len(intersection)
    denominator = min(len(file_list_1), len(file_list_2))
    # Issues with only one file in file_list create expressions of type "0 / 1" or "1 / 1", and then, if the file
    # exists in the file_list of the other issues, mutual is 1 even if the other issue have 20 files.
    if denominator == 1:
        if abs(len(file_list_1) - len(file_list_2)) > 1:
            numerator = 0
    if denominator > 0:
        mutual = numerator / denominator

    return mutual


def jaccard_score(file_list_1: list = None, file_list_2: list = None) -> float:
    result = 0.0
    intersection = list(set(file_list_1).intersection(file_list_2))
    union = list(set(file_list_1).union(file_list_2))

    if len(union) > 0:
        result = len(intersection) / len(union)

    return result


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    log.info(f"Finding similar issues ...")

    # MongoDB client.
    mongodb_client: MongoClient = get_default_mongo_client()

    # Load Gerrit data.
    db = mongodb_client[os.environ.get('GERRIT_DB_NAME', input_params['db_name'])]
    collection = db[input_params['collection_name']]
    data = collection.find({}, {'_id': 0})
    df = pd.DataFrame(list(data))

    # Check empty Dataframe.
    if 0 == df.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{db.name}.{collection.name}' collection.")

    # Initialize empty list column.
    df['sim_bugs'] = [[] for i in range(len(df))]
    df['near_bugs'] = [[] for i in range(len(df))]

    # Iterate over all rows.
    all_sim_bugs = []
    all_near_bugs = []
    for i in tqdm(range(df.shape[0])):
        # Compare each row with all rows.
        for j in range(df.shape[0]):
            if j == i:
                continue
            bug_anc = df.at[i, 'bug_id']
            bug_pos = df.loc[j, 'bug_id']
            jaccard_similarity = jaccard_score(df.loc[i, 'file_list'], df.loc[j, 'file_list'])
            if jaccard_similarity >= float(input_params['similarity_threshold']):
                if [bug_anc, bug_pos] not in all_sim_bugs and [bug_pos, bug_anc] not in all_sim_bugs:
                    df.at[i, 'sim_bugs'].append(int(bug_pos))
                    all_sim_bugs.append([bug_anc, bug_pos])

            if float(input_params['similarity_threshold']) - 0.25 <= jaccard_similarity < \
                    float(input_params['similarity_threshold']):
                if [bug_anc, bug_pos] not in all_near_bugs and [bug_pos, bug_anc] not in all_near_bugs:
                    df.at[i, 'near_bugs'].append(int(bug_pos))
                    all_near_bugs.append([bug_anc, bug_pos])

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df.shape[0] / input_params['batch_size'])
    batches = np.array_split(df, num_batches)

    output_collection = db[input_params['output_collection_name']]

    # Drop collection if already exists.
    if output_collection.name in db.list_collection_names():
        log.info(f"Dropping collection '{db.name}.{output_collection.name}' ...")
        db.drop_collection(output_collection.name)

    inserted_docs_number = 0
    for batch in batches:
        log.info(f"Inserting documents ...")
        inserted_documents = output_collection.insert_many(batch.to_dict("records"))
        inserted_docs_number += len(inserted_documents.inserted_ids)

    log.info(f"Inserted documents: {inserted_docs_number}")

    final_time = time.time()
    log.info(f"Finding similar issues total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
