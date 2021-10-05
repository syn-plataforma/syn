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

    parser.add_argument('--input_db_name', default='bugzilla', type=str, help='Database name.')
    parser.add_argument('--original_collection_name', default='normalized_clear', type=str, help='Collection name.')
    parser.add_argument('--input_collection_name', default='pairs_first_step', type=str, help='Collection name.')
    parser.add_argument('--output_collection_name', default='pairs', type=str, help='Pairs collection name.')
    parser.add_argument('--batch_size', default=10000, type=str, help='Batch size.')

    args = parser.parse_args()

    return {
        'input_db_name': args.input_db_name,
        'original_collection_name': args.original_collection_name,
        'input_collection_name': args.input_collection_name,
        'output_collection_name': args.output_collection_name,
        'batch_size': args.batch_size
    }


def add_duplicate_pair(bug1, bug2, dup_issues, no_dup_bug_id_list, no_dup_issues, pairs):
    # Add duplicate pair.
    if [bug1, bug2] not in dup_issues and [bug2, bug1] not in dup_issues:
        dup_issues.append([bug1, bug2])
        pairs.append(
            {
                'bug1': bug1,
                'bug2': bug2,
                'dec': 1
            }
        )

    # Add no similar pair.
    bug3 = int(random.choice(no_dup_bug_id_list))
    no_dup_bug_id_list.pop(no_dup_bug_id_list.index(bug3))
    if [bug1, bug3] not in no_dup_issues and [bug3, bug1] not in no_dup_issues:
        no_dup_issues.append([bug1, bug3])
        pairs.append(
            {
                'bug1': bug1,
                'bug2': bug3,
                'dec': -1
            }
        )

    return dup_issues, no_dup_issues, pairs


def check_indirect_relations(
        df_pairs: pd.DataFrame = None,
        no_duplicate_bug_id_list: list = None
) -> list:
    # Initialize all_pairs
    all_pairs = []
    for ind in df_pairs.index:
        all_pairs.append([df_pairs['bug1'][ind], df_pairs['bug2'][ind]])

    # Initialize all_duplicate_issues
    df_duplicates = df_pairs.loc[df_pairs['dec'] == 1].copy()
    all_duplicate_issues = []
    for ind in df_duplicates.index:
        all_duplicate_issues.append([df_duplicates['bug1'][ind], df_duplicates['bug2'][ind]])

    # Initialize all_no_duplicate_issues
    df_no_duplicates = df_pairs.loc[df_pairs['dec'] == 0].copy()
    all_no_duplicate_issues = []
    for ind in df_no_duplicates.index:
        all_no_duplicate_issues.append([df_no_duplicates['bug1'][ind], df_no_duplicates['bug2'][ind]])

    for i in tqdm(range(df_pairs.shape[0])):
        bug1 = int(df_pairs['bug1'][i])
        bug2 = int(df_pairs['bug2'][i])

        for ind in df_pairs.index:
            if df_pairs['bug1'][ind] != bug1:
                if df_pairs['bug1'][ind] == bug2:
                    all_duplicate_issues, all_no_duplicate_issues, all_pairs = add_duplicate_pair(
                        bug1=bug1,
                        bug2=df_pairs['bug2'][ind],
                        dup_issues=all_no_duplicate_issues,
                        no_dup_bug_id_list=no_duplicate_bug_id_list,
                        no_dup_issues=all_no_duplicate_issues,
                        pairs=all_pairs
                    )

                if df_pairs['bug2'][ind] == bug2:
                    all_duplicate_issues, all_no_duplicate_issues, all_pairs = add_duplicate_pair(
                        bug1=bug1,
                        bug2=df_pairs['bug1'][ind],
                        dup_issues=all_no_duplicate_issues,
                        no_dup_bug_id_list=no_duplicate_bug_id_list,
                        no_dup_issues=all_no_duplicate_issues,
                        pairs=all_pairs
                    )

    return all_pairs


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    log.info(f"Finding duplicate issues ...")

    # MongoDB client.
    mongodb_client: MongoClient = get_default_mongo_client()

    # Load Bugzilla data.
    db = mongodb_client[input_params['input_db_name']]

    # Collection all issues (duplicated and no duplicated).
    original_collection = db[input_params['original_collection_name']]
    query = {}
    projection = {'_id': 0, 'bug_id': 1}
    log.info(f"Reading data from '{db.name}.{original_collection.name}' ...")
    original_data = original_collection.find(query, projection)
    df_original = pd.DataFrame(list(original_data))

    # Check empty Dataframe.
    if 0 == df_original.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{db.name}.{original_collection.name}' collection.")

    # List with all bug_id.
    original_bug_id_list = df_original['bug_id'].to_list()

    collection_pairs_first_step = db[input_params['input_collection_name']]
    query = {}
    projection = {'_id': 0}
    data_pairs_first_step = collection_pairs_first_step.find(query, projection)
    df_pairs_first_step = pd.DataFrame(list(data_pairs_first_step))

    # Check empty Dataframe.
    if 0 == df_pairs_first_step.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{db.name}.{collection_pairs_first_step.name}' collection.")

    bug1_list = df_pairs_first_step['bug1'].to_list()
    bug2_list = df_pairs_first_step['bug2'].to_list()

    no_duplicate_bug_id_list = set(original_bug_id_list) - set(bug1_list) - set(bug2_list)

    # For each issue that have similar issues generate a pair of similar issues and a pair of no similar issues.
    pairs_from_indirect_relations = check_indirect_relations(df_pairs_first_step, list(no_duplicate_bug_id_list))
    log.info(f"Pairs generated: {len(pairs_from_indirect_relations)}")

    # Dataframe pairs.
    df_pairs = pd.DataFrame(pairs_from_indirect_relations)

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df_pairs.shape[0] / input_params['batch_size'])
    batches = np.array_split(df_pairs, num_batches)

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
