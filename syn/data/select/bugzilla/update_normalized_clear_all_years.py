#!/usr/bin/env python3

import argparse
import os
import time

from pymongo import MongoClient

from syn.helpers.argparser import common_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[common_parser],
        description='Retrieve clear and normalized clear collections.'
    )

    parser.add_argument('--no_drop_collection', default=False, dest='drop_collection', action='store_false',
                        help="No drop previous normalized_clear collection.")
    parser.add_argument('--drop_collection', dest='drop_collection', action='store_true',
                        help="Drop previous normalized_clear collection.")
    parser.add_argument('--closed_states', default=True, dest='closed_states', action='store_true',
                        help="Filter by state CLOSED, RESOLVED, VERIFIED.")
    parser.add_argument('--no_closed_states', dest='closed_states', action='store_false', help="No filter by state.")

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'closed_states': args.closed_states,
        'drop_collection': args.drop_collection,
        'mongo_batch_size': args.mongo_batch_size
    }


def get_command(
        os_name: str = 'posix',
        corpus: str = 'openOffice',
        year: int = 2000,
        closed_states: bool = True
):
    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    arguments = f" -m syn.data.select.bugzilla.update_normalized_clear_year --corpus {corpus} --year {year}"
    if not closed_states:
        arguments = f"{arguments} --no_closed_states"
    cmd = {
        'posix': f"python3 {arguments}",
        'nt': f"python {arguments}",
        'java': ''
    }

    return cmd[os_name]


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    log.info(f"Updating all normalized_clear years ...")

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params['corpus']]
    col_name = 'normalized_clear_updated' if input_params['closed_states'] else 'normalized_clear_all_states'
    col = db[col_name]

    if input_params['drop_collection'] and col.name in db.list_collection_names():
        log.info(f"Dropping collection '{db.name}.{col.name}'")
        db.drop_collection(col.name)

    # Defines Python executable.
    for year in range(2000, 2021):
        cmd = get_command(os.name, input_params['corpus'], year, input_params['closed_states'])

        # Run command.
        log.info(f"Running command: '{cmd}'")
        os.system(cmd)

    final_time = time.time()
    log.info(f"Updating all normalized_clear years total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
