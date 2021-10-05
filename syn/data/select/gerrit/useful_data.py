#!/usr/bin/env python3
import argparse
import math
import os
import re
import time
from typing import Union

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
    parser = argparse.ArgumentParser(description='Search useful issues.')

    parser.add_argument('--db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--collection_name', default='eclipse', type=str, help='Gerrit collection name.')
    parser.add_argument('--output_collection_name', default='eclipse_useful', type=str,
                        help='Gerrit output collection name.')
    parser.add_argument('--batch_size', default=10000, type=int, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_collection_name': args.output_collection_name,
        'batch_size': args.batch_size
    }


def get_filtered_file_list(files: Union[list, pd.DataFrame] = None) -> list:
    result = []
    if isinstance(files, list) and len(files) > 0:
        for file in files:
            for excluded_file in os.environ['GERRIT_EXCLUDED_FILES'].split(','):
                if file.find(excluded_file) == -1:
                    for extension in os.environ['GERRIT_EXCLUDED_EXTENSIONS'].split(','):
                        if file.find(extension) == -1:
                            result.append(file)

    return result


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    log.info(f"Processing Gerrit issues ...")

    # MongoDB data.
    mongodb_client: MongoClient = get_default_mongo_client()
    db = mongodb_client[os.environ.get('GERRIT_DB_NAME', input_params['db_name'])]
    col = db[input_params['collection_name']]
    data = col.find({}, {'_id': 0})
    df = pd.DataFrame(list(data))
    df['bug_id'] = -1
    # Search for bugs that don't matches this pattern.
    bug_id_list = []
    loop = tqdm(range(df.shape[0]))
    for i in loop:
        matched_1 = re.search(r'\[\s*\+?(-?\d+)\s*]', df.loc[i, 'subject'])
        matched_2 = re.search(r'[Bb][Uu][Gg]\s[0-9]+', df.loc[i, 'subject'])
        is_match = bool(matched_1) + bool(matched_2)
        if bool(is_match):
            flag = True
            res = re.findall(r'[Bb][Uu][Gg]\s[0-9]+', df.iloc[i]['subject'])
            if not res:
                flag = False
                res = re.findall(r'\[\s*\+?(-?\d+)\s*]', df.iloc[i]['subject'])
            bug_id = int(res[0][4:]) if flag else int(res[0])
            if bug_id in bug_id_list:
                idx = df.index[df['bug_id'] == bug_id].tolist()
                if len(idx) > 1:
                    raise ValueError(f"There are more than one bug with 'bug_id' = {bug_id}")
                previous_file_list = get_filtered_file_list(df.iloc[idx[0]]['file_list'])
                current_file_list = get_filtered_file_list(df.iloc[i]['file_list'])
                file_list = set(previous_file_list + current_file_list)
                df.at[idx[0], 'file_list'] = list(file_list)
            else:
                bug_id_list.append(bug_id)
                df.at[i, 'bug_id'] = bug_id
                df.at[i, 'file_list'] = get_filtered_file_list(df.iloc[i]["file_list"])

    # Drop collection if already exists.
    if input_params['output_collection_name'] in db.list_collection_names():
        log.info(f"Dropping collection {input_params['output_collection_name']} ...")
        db.drop_collection(input_params['output_collection_name'])

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(df.shape[0] / input_params['batch_size'])
    batches = np.array_split(df.loc[df['bug_id'] != -1], num_batches)

    # Insert documents with bug_id in MongoDB.
    inserted_docs_number = 0
    for batch in batches:
        log.info(f"Inserting documents ...")
        inserted_documents = db[input_params['output_collection_name']].insert_many(batch.to_dict("records"))
        inserted_docs_number += len(inserted_documents.inserted_ids)

    final_time = time.time()
    log.info(f"Processing Gerrit issues total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
