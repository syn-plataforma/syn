#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.mongodb import get_default_mongo_client, get_default_local_mongo_client


def get_raw_data_query_and_projection():
    fields = {
        "dec": 1,
        "constituency_tree_description1": 1,
        "constituency_tree_description2": 1,
        "detailed_tokens1": 1,
        "detailed_tokens2": 1,
        "bug1": 1,
        "bug2": 1,
    }

    query = {
        "constituency_tree_description1": {"$exists": True},
        "constituency_tree_description2": {"$exists": True},
        "detailed_tokens1.0": {'$exists': True},
        "detailed_tokens2.0": {'$exists': True},
    }

    return query, fields


def get_raw_data_query_and_projection_categorical(column=None):
    fields = {
        column: 1,
        "constituency_trees": 1,
        "detailed_tokens": 1,
        "bug_id": 1,
    }

    query = {
        "constituency_trees": {"$exists": True},
        "detailed_tokens.0": {'$exists': True},
    }

    return query, fields


def get_constituency_tree_raw_data(db_name, col_name, categorical=True):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()
    # Define el logger que se utilizará.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    log.debug(f"\n[INICIO EJECUCIÓN]")

    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    mongodb_client: MongoClient = get_default_mongo_client() if os.environ['WORK_ENVIRONMENT'] == 'aws' \
        else get_default_local_mongo_client()

    db = mongodb_client[db_name]
    col = db[col_name]

    if not categorical:
        query, fields = get_raw_data_query_and_projection()
    else:
        query, fields = get_raw_data_query_and_projection_categorical()

    return col.find(query, fields)
