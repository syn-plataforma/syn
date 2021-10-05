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
    eclipse_base_col = db["eclipse_base"]

    # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    fields = {
        "_id": 0
    }

    eclipse_base_data = eclipse_base_col.find({}, fields)
    # .limit(20)

    # Expande el cursor y construye el DataFrame
    eclipse_base = pd.DataFrame(list(eclipse_base_data))

    eclipse_base_summary_col = db["clear_description_nlp_2"]
    eclipse_base_summary_data = eclipse_base_summary_col.find({}, fields)
    eclipse_base_summary = pd.DataFrame(list(eclipse_base_summary_data))

    eclipse_base_data_description_col = db["clear_short_desc_nlp"]
    eclipse_base_data_description_data = eclipse_base_data_description_col.find({}, fields)
    eclipse_base_data_description = pd.DataFrame(list(eclipse_base_data_description_data))

    # Merge de los dataframes
    eclipse_base_summary_description = pd.merge(eclipse_base_summary, eclipse_base_data_description, on='id')
    eclipse_base_clear = pd.merge(eclipse_base, eclipse_base_summary_description, on='id')

    # Almacena el dataframe en MongoDB.
    db["eclipse_base_clear"].insert_many(eclipse_base_clear.to_dict('records'))

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
