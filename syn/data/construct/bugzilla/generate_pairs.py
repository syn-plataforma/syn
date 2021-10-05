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

    parser.add_argument('--db_name', default='bugzilla', type=str, help='Database name.')
    parser.add_argument('--collection_name', default='normalized_clear', type=str, help='Collection name.')
    parser.add_argument('--output_collection_name', default='pairs_first_step', type=str, help='Pairs collection name.')
    parser.add_argument('--batch_size', default=10000, type=str, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_collection_name': args.output_collection_name,
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
        dup_issues = []
        if not math.isnan(df.loc[i, column_name]):
            dup_issues.append(int(df.loc[i, column_name]))
        no_dup_issues = list(set(bug_id_list) - set(dup_issues) - {bug1})
        if not math.isnan(df.loc[i, column_name]):
            for dup in dup_issues:
                # Add similar pair.
                bug2 = int(dup)
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
                bug3 = int(random.choice(no_dup_issues))
                no_dup_issues.pop(no_dup_issues.index(bug3))
                if [bug1, bug3] not in all_no_similar_issues and [bug3, bug1] not in all_no_similar_issues:
                    all_no_similar_issues.append([bug1, bug3])
                    all_pairs.append(
                        {
                            'bug1': bug1,
                            'bug2': bug3,
                            'dec': -1
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

    log.info(f"Finding duplicate issues ...")

    # MongoDB client.
    mongodb_client: MongoClient = get_default_mongo_client()

    # Load Gerrit data.
    db = mongodb_client[os.environ.get('BUGZILLA_MONGODB_DATABASE_NAME', input_params['db_name'])]
    collection = db[input_params['collection_name']]
    query = {}
    projection = {'_id': 0, 'bug_id': 1, 'dup_id': 1}
    log.info(f"Reading data from '{db.name}.{collection.name}' ...")
    data = collection.find(query, projection)
    df = pd.DataFrame(list(data))

    # Check empty Dataframe.
    if 0 == df.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{db.name}.{collection.name}' collection.")

    # For each issue that have similar issues generate a pair of similar issues and a pair of no similar issues.
    pairs_first_step = generate_pairs(df, 'dup_id')
    log.info(f"Pairs generated: {len(pairs_first_step)}")

    # Dataframe pairs.
    df_pairs = pd.DataFrame(pairs_first_step)

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
