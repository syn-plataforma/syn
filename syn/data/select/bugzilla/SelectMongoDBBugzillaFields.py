#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.system import check_same_python_module_already_running


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()
    # Define el logger que se utilizará.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Carga el fichero de configuración para el entorno.
    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    # Inicializa los parámetros MongoDB para almacenar las estadísticas.
    mongodb_client: MongoClient = get_default_mongo_client()
    db = mongodb_client["bugzilla"]
    col = db["eclipse_all"]

    # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    fields = {
        "_id": 0,
        "assigned_to": 1,
        "assigned_to_detail": 1,
        "classification": 1,
        "component": 1,
        "creation_time": 1,
        "creator": 1,
        "creator_detail": 1,
        "dupe_of": 1,
        "id": 1,
        "op_sys": 1,
        "platform": 1,
        "priority": 1,
        "product": 1,
        "resolution": 1,
        "severity": 1,
        "status": 1,
        "summary": 1,
        "version": 1,
        "description": "$comments.text"
    }
    aggregation_project = {"$project": fields}

    # Se utilizarán sólo las incidencias resueltas.

    aggregation_match = {"$match": {
        '$and': [
            {
                '$or': [
                    {'status': {'$eq': 'RESOLVED'}},
                    {'status': {'$eq': 'VERIFIED'}},
                    {'status': {'$eq': 'CLOSED'}}
                ]
            },
            {'comments': {'$exists': 'true'}},
            {'comments': {'$ne': 'null'}},
            {'comments': {'$ne': ""}},
            {'comments': {'$not': {'$size': 0}}},
            {'comments.count': {'$eq': 0}}
        ]
    }}
    aggregation_unwind = {"$unwind": "$comments"}
    aggregation_limit = {"$limit": 20}

    # data = col.find(query, fields).limit(20)
    data = col.aggregate([
        aggregation_unwind,
        aggregation_match,
        # aggregation_limit,
        aggregation_project
    ])

    # Expande el cursor y construye el DataFrame
    eclipse_base = pd.DataFrame(list(data))

    # Almacena el dataframe en MongoDB.
    db["eclipse_base"].insert_many(eclipse_base.to_dict('records'))

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
