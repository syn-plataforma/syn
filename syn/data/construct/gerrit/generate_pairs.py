#!/usr/bin/env python3
import argparse
import math
import os
import random
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
    parser = argparse.ArgumentParser(description='Generate pairs for similar issues.')

    parser.add_argument('--db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--collection_name', default='eclipse_similarities', type=str,
                        help='Gerrit similarities collection name.')
    parser.add_argument('--output_db_name', default='bugzilla', type=str,
                        help='Gerrit pairs database name.')
    parser.add_argument('--output_similar_collection_name', default='similar_pairs', type=str,
                        help='Gerrit similar pairs collection name.')
    parser.add_argument('--output_near_collection_name', default='near_pairs', type=str,
                        help='Gerrit near pairs collection name.')
    parser.add_argument('--no_near_issues', default=False, dest='near_issues', action='store_false',
                        help="No near issues.")
    parser.add_argument('--near_issues', dest='near_issues', action='store_true', help="Near issues.")
    parser.add_argument('--batch_size', default=10000, type=str, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_db_name': args.output_db_name,
        'output_similar_collection_name': args.output_similar_collection_name,
        'output_near_collection_name': args.output_near_collection_name,
        'near_issues': args.near_issues,
        'batch_size': args.batch_size
    }


def generate_pairs(df: pd.DataFrame = None, column_name: str = 'sim_bugs') -> list:
    all_pairs = []
    all_similar_issues = []
    all_no_similar_issues = []

    # List with all bug_id.
    bug_id_list = df['bug_id'].to_list()
    for i in tqdm(range(df.shape[0])):
        bug1 = int(df.loc[i, 'bug_id'])
        similar_issues = df.loc[i, column_name]
        no_similar_issues = list(set(bug_id_list) - set(similar_issues) - {bug1})
        if len(similar_issues) > 0:
            for sim in similar_issues:
                # Add similar pair.
                bug2 = int(sim)
                if [bug1, bug2] not in all_similar_issues and [bug2, bug1] not in all_similar_issues:
                    all_similar_issues.append([bug1, bug2])
                    all_pairs.append(
                        {
                            'bug1': bug1,
                            'bug2': bug2,
                            'dec': 1
                        }
                    )

                # Add no similar pair.
                bug3 = int(random.choice(no_similar_issues))
                no_similar_issues.pop(no_similar_issues.index(bug3))
                if [bug1, bug3] not in all_no_similar_issues and [bug3, bug1] not in all_no_similar_issues:
                    all_no_similar_issues.append([bug1, bug3])
                    all_pairs.append(
                        {
                            'bug1': bug1,
                            'bug2': bug3,
                            'dec': 0
                        }
                    )

    return all_pairs


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

    similar_column_name = 'sim_bugs' if not input_params['near_issues'] else 'near_bugs'
    # For each issue that have similar issues generate a pair of similar issues and a pair of no similar issues.
    pairs = generate_pairs(df, similar_column_name)
    log.info(f"Pairs generated: {len(pairs)}")

    # Dataframe pairs.
    df_pairs = pd.DataFrame(pairs)

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df_pairs.shape[0] / input_params['batch_size'])
    batches = np.array_split(df_pairs, num_batches)

    output_db = mongodb_client[input_params['output_db_name']]
    output_collection = output_db[input_params['output_similar_collection_name']] if not input_params['near_issues'] \
        else output_db[input_params['output_near_collection_name']]

    # Drop collection if already exists.
    if output_collection.name in output_db.list_collection_names():
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
