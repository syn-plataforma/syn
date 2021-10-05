#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.mongodb import get_default_mongo_client, get_default_local_mongo_client
from syn.model.build.treelstm.vectorizer.ConstituencyParser import get_raw_data_query_and_projection, \
    get_raw_data_query_and_projection_categorical


def get_attention_vector_raw_data(db_name, col_name, categorical=True, column=None):
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
        fields["constituents_embeddings_description1"] = 1
        fields["constituents_embeddings_description2"] = 1
        query["constituents_embeddings_description1.0"] = {'$exists': True}
        query["constituents_embeddings_description2.0"] = {'$exists': True}
    else:
        query, fields = get_raw_data_query_and_projection_categorical(column=column)
        fields["constituents_embeddings"] = 1
        query["constituents_embeddings.0"] = {'$exists': True}

    return col.find(query, fields)
