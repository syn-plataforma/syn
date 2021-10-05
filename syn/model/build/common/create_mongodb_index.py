#!/usr/bin/env python3
import argparse
import os
import time

from pymongo import MongoClient, DESCENDING, ASCENDING

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Create MongoDB index.')

    parser.add_argument('--db_name', default='bugzilla', type=str, help='Database name.')
    parser.add_argument('--collection_name', default='normalized_clear', type=str, help='Collection name.')
    parser.add_argument('--field_name', default='creation_ts', type=str, help='Index field.')
    parser.add_argument('--order', default=-1, type=int, choices=[1, -1], help='Ascending: 1. Descending: -1.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'field_name': args.field_name,
        'order': args.order
    }


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    order = 'ascending' if input_params['order'] == 1 else 'descending'
    log.info(f"Creating {order} index on field: '{input_params['db_name']}.{input_params['collection_name']}."
             f"{input_params['field_name']}' ...")

    # MongoDB client.
    mongodb_client: MongoClient = get_default_mongo_client()
    db = mongodb_client[input_params['db_name']]
    col = db[input_params['collection_name']]
    col.create_index([(input_params['field_name'], DESCENDING if input_params['order'] == -1 else ASCENDING)])

    final_time = time.time()
    log.info(f"Creating {order} index on field: '{input_params['field_name']}' total execution time = "
             f"{((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
