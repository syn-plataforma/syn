#!/usr/bin/env python3

import argparse
import os
import time

import pandas as pd

from syn.helpers.argparser import common_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[common_parser],
        description='Retrieve clear and normalized clear collections.'
    )

    parser.add_argument('--year', default=2000, type=int, help='Reference year.')

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'year': args.year,
        'mongo_batch_size': args.mongo_batch_size
    }


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    log.info(f"Updating normalized_clear for year '{input_params['year']}' ...")

    # Load clear collection to get "assigned_to" field.
    query = {"bug_status": "UNCONFIRMED"}
    projection = {'_id': 0, 'bug_id': 1}
    df_clear = load_dataframe_from_mongodb(
        database_name=input_params['corpus'],
        collection_name='clear',
        query=query,
        projection=projection
    )
    log.info(f"df_clear: {df_clear.shape[0]}")

    # Load normalized_clear collection to get add "assigned_to" field.
    df_normalized_clear_updated = load_dataframe_from_mongodb(
        database_name=input_params['corpus'],
        collection_name='normalized_clear_updated',
        query=query,
        projection=projection
    )
    log.info(f"normalized_clear_updated: {df_normalized_clear_updated.shape[0]}")

    # Join on column 'bug_id'.
    joined_df = df_clear.merge(
        df_normalized_clear_updated,
        left_on='bug_id',
        right_on='bug_id',
        how='left',
        indicator=True
    )

    print(joined_df.loc[joined_df['_merge'] == 'left_only'])
    log.info(f"normalized_clear_updated: {joined_df.shape[0]}")

    final_time = time.time()
    log.info(f"Updating normalized_clear year total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
