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
    parser = argparse.ArgumentParser(description='Find similar issues.')

    parser.add_argument('--db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--collection_name', default='eclipse_similarities', type=str,
                        help='Gerrit similarities collection name.')
    parser.add_argument('--output_collection_name', default='eclipse_triplets', type=str,
                        help='Gerrit triplets collection name.')
    parser.add_argument('--batch_size', default=10000, type=str, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_collection_name': args.output_collection_name,
        'batch_size': args.batch_size
    }


def generate_triplets_for_similar_issues(
        df: pd.DataFrame = None,
        bug_id_list: list = None,
) -> list:
    triplets_for_similar_issues = []
    all_similar_issues = []
    for i in tqdm(range(df.shape[0])):
        similar_issues = df.loc[i, 'sim_bugs']
        current_issue = df.loc[i, 'bug_id']
        no_similar_issues = list(set(bug_id_list) - set(similar_issues) - set(list(current_issue)))
        for sim in similar_issues:
            bug_anc = int(current_issue)
            bug_pos = int(sim)
            if [bug_anc, bug_pos] not in all_similar_issues and [bug_pos, bug_anc] not in all_similar_issues:
                all_similar_issues.append([bug_anc, bug_pos])
                bug_neg = int(random.choice(no_similar_issues))
                no_similar_issues.pop(no_similar_issues.index(bug_neg))
                triplets_for_similar_issues.append(
                    {
                        'bug_anc': bug_anc,
                        'bug_pos': bug_pos,
                        'bug_neg': bug_neg,
                        'dec': 1
                    }
                )

    return triplets_for_similar_issues


def generate_triplets_for_no_similar_issues(
        df: pd.DataFrame = None,
        bug_id_list: list = None,
        max_triplets_number: int = 0
) -> list:
    triplets_for_no_similar_issues = []
    all_no_similar_issues = []
    if max_triplets_number == 0:
        max_triplets_number = df.shape[0]
    while max_triplets_number >= len(triplets_for_no_similar_issues):
        for i in tqdm(range(df.shape[0])):
            current_issue = df.loc[i, 'bug_id']
            no_similar_issues = list(set(bug_id_list) - set(list(current_issue)))
            bug_anc = int(current_issue)
            bug_pos = int(random.choice(no_similar_issues))
            no_similar_issues.pop(no_similar_issues.index(bug_pos))
            if [bug_anc, bug_pos] not in all_no_similar_issues and [bug_pos, bug_anc] not in all_no_similar_issues:
                all_no_similar_issues.append([bug_anc, bug_pos])
                bug_neg = int(random.choice(no_similar_issues))
                triplets_for_no_similar_issues.append(
                    {
                        'bug_anc': bug_anc,
                        'bug_pos': bug_pos,
                        'bug_neg': bug_neg,
                        'dec': 0
                    }
                )

    return triplets_for_no_similar_issues


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

    # List with all bug_id.
    bug_id_list = df['bug_id'].to_list()

    # For each issue that have similar issues generate a triplet of similar issues.
    triplets_for_similar_issues = generate_triplets_for_similar_issues(
        df[len(df['sim_bugs']) > 0],
        bug_id_list
    )
    log.info(f"Triplets for similar issues: {len(triplets_for_similar_issues)}")

    # For each issue that not have similar issues generate a triplet of no similar issues.
    triplets_for_no_similar_issues = generate_triplets_for_no_similar_issues(
        df[len(df['sim_bugs']) == 0],
        bug_id_list,
        len(triplets_for_similar_issues)
    )
    log.info(f"Triplets for no similar issues: {len(triplets_for_no_similar_issues)}")

    # Concatenate lists.
    triplets = triplets_for_similar_issues + triplets_for_no_similar_issues

    # Dataframe triplets.
    df_triplets = pd.DataFrame(triplets)

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df_triplets.shape[0] / input_params['batch_size'])
    batches = np.array_split(df_triplets, num_batches)

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
