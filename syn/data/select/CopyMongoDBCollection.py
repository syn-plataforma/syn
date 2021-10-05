#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.mongodb import get_default_mongo_client, get_input_params
from syn.helpers.system import check_same_python_module_already_running


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()
    # Define el logger que se utilizará.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Incializa las variables que almacenarán los argumentos de entrada.
    input_params = get_input_params()

    # Establece los parámetros que se enviarán en la petición.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[input_params.db_name]
    source_collection = db[input_params.collection_name]
    target_collection = db[f"{input_params.collection_name}_embeddings"]

    log.debug(f"Colecciones exitentes en '{db.name}': {str(db.list_collection_names())}")

    if target_collection.name in db.list_collection_names() and input_params.drop_collection:
        db.drop_collection(target_collection.name)

    cursor = source_collection.find(
        {},
        {
            "creation_ts": "$creation_time",
            "short_desc": "$summary",
            "bug_status": "$status",
            "bug_id": "$id",
            "dup_id": "$dupe_of",
            "resolution": 1,
            "version": 1,
            "product": 1,
            "priority": 1,
            "component": 1,
            "delta_ts": 1,
            "bug_severity": "$severity",
            "description": "$comments.0",
            "normalized_short_desc": 1,
            "normalized_description": 1,
            "comments": {"$slice": 1}
        }
    )

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
