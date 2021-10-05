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
    db = mongodb_client["eclipse"]
    clear_col = db["clear"]

    # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    fields = {
        "_id": 0,
    }

    # Se utilizarán sólo las incidencias resueltas.
    query = {
        '$or': [
            {'bug_status': {'$eq': 'RESOLVED'}},
            {'bug_status': {'$eq': 'VERIFIED'}}
        ]
    }

    clear_data = clear_col.find(query, fields)
    # .limit(20)

    # Expande el cursor y construye el DataFrame
    clear = pd.DataFrame(list(clear_data))

    clear_short_desc_nlp_col = db["clear_short_desc_nlp"]
    clear_short_desc_nlp_data = clear_short_desc_nlp_col.find({}, fields)
    clear_short_desc_nlp = pd.DataFrame(list(clear_short_desc_nlp_data))

    clear_description_nlp_col = db["clear_description_nlp_2"]
    clear_description_nlp_data = clear_description_nlp_col.find({}, fields)
    clear_description_nlp = pd.DataFrame(list(clear_description_nlp_data))

    # Merge de los dataframes
    clear_short_desc_description_nlp = pd.merge(clear_short_desc_nlp, clear_description_nlp, on='bug_id')
    clear_nlp = pd.merge(clear, clear_short_desc_description_nlp, on='bug_id')

    # Almacena el dataframe en MongoDB.
    db["clear_nlp"].insert_many(clear_nlp.to_dict('records'))

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
