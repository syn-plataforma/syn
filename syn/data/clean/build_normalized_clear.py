#!/usr/bin/env python3

import argparse
import math
import os
import time

import numpy as np
from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.nlp.TextNormalizer import normalize_incidence, get_codebooks_tokens, count_tokens
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Search useful issues.')

    parser.add_argument('--db_name', default='bugzilla', type=str, help='Database name.')
    parser.add_argument('--collection_name', default='clear', type=str, help='Clear collection name.')
    parser.add_argument('--output_collection_name', default='normalized_clear', type=str,
                        help='Normalized clear collection name.')
    parser.add_argument('--mongo_batch_size', default=10000, type=str, help='Batch size.')
    parser.add_argument('--max_num_tokens', default=os.environ.get('EMBEDDING_MONGODB_MAX_NUM_TOKENS'), type=int,
                        help='Maximum number of tokens in the text.')
    parser.add_argument('--architecture', default='tree_lstm', type=str, help='Architecture of de solution.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'output_collection_name': args.output_collection_name,
        'mongo_batch_size': args.mongo_batch_size,
        'max_num_tokens': args.max_num_tokens,
        'architecture': args.architecture
    }


if __name__ == "__main__":
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    log.info(f"Building 'normalized_clear' collection ...")

    # Load clear collection.
    df_clear = load_dataframe_from_mongodb(
        database_name=input_params['db_name'],
        collection_name=input_params['collection_name']
    )

    # Check empty Dataframe.
    if 0 == df_clear.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{input_params['db_name']}.{input_params['collection_name']}' collection.")

    df_normalized_clear = df_clear.copy()

    # Normalize short description.
    log.info("Normalizing short description ...")
    tic = time.time()
    df_normalized_clear['normalized_short_desc'] = df_normalized_clear['short_desc'].apply(
        lambda x: normalize_incidence(x, to_lower_case=True)
    )
    log.info(f"Normalizing short description total execution time = {((time.time() - tic) / 60)} minutes")

    # Normalize description.
    log.info("Normalizing description ...")
    tic = time.time()
    df_normalized_clear['normalized_description'] = df_normalized_clear['description'].apply(
        lambda x: normalize_incidence(x, to_lower_case=True)
    )
    log.info(f"Normalizing description total execution time = {((time.time() - tic) / 60)} minutes")

    # add column with tokens and with number off tokens
    log.info("Adding tokens and number of tokens of normalized description ...")
    tic = time.time()
    df_normalized_clear['detailed_tokens'] = df_normalized_clear['normalized_description'].apply(
        lambda x: get_codebooks_tokens(x)
    )
    df_normalized_clear['total_num_tokens'] = df_normalized_clear['detailed_tokens'].apply(
        lambda x: count_tokens(x)
    )
    log.info(f"Adding number of tokens of normalized description "
             f"total execution time = {((time.time() - tic) / 60)} minutes")

    # filter by number of tokens
    if input_params['max_num_tokens'] > 0:
        log.info("Filtering by number of tokens of normalized description ...")
        tic = time.time()
        result = df_normalized_clear[df_normalized_clear['total_num_tokens'] <= input_params['max_num_tokens']].copy()
        log.info(f"Filtering by number of tokens of normalized description "
                 f"total execution time = {((time.time() - tic) / 60)} minutes")
    else:
        log.info("No filter by  number of tokens of normalized description applied.")
        result = df_normalized_clear.copy()

    # drop columns not needed in Tree-LSTM architecture
    if input_params['architecture'] == 'tree_lstm':
        result.drop(['detailed_tokens', 'total_num_tokens'], axis=1, inplace=True)

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params['db_name']]
    col = db[input_params['output_collection_name']]

    # Si existe una versión previa de la colección MongoDB la elimina.
    if col.name in db.list_collection_names():
        log.info(f"Dropping collection: '{col.name}'")
        db.drop_collection(col.name)

    # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
    num_batches = math.ceil(result.shape[0] / input_params['mongo_batch_size'])
    batches = np.array_split(result, num_batches)

    log.info("Inserting documents in MongoDB ...")
    tic = time.time()
    inserted_docs_number = 0
    for batch in batches:
        log.info(f"Inserting documents ...")
        a = batch.to_dict("records")
        inserted_documents = col.insert_many(batch.to_dict("records"))
        inserted_docs_number += len(inserted_documents.inserted_ids)

    log.info(f"Inserting documents total execution time = {((time.time() - tic) / 60)} minutes")
    log.info(f"Inserted documents: {inserted_docs_number}")

    final_time = time.time()
    log.info(
        f"Building 'normalized_clear' collection total execution time = {((final_time - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
