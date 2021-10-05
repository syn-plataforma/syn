#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.jira.JiraHelper import get_issues_by_date_range, get_input_params
from syn.helpers.mongodb import save_issues_to_mongodb, get_default_mongo_client
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

    db = mongodb_client[input_params['mongo_params'].db_name]
    col = db[
        f"{input_params['mongo_params'].collection_name}"
        f"_{input_params['jira_api_params'].year}_{input_params['jira_api_params'].year + 1}"
    ]

    log.debug(f"Colecciones exitentes en '{db.name}': {str(db.list_collection_names())}")

    if col.name in db.list_collection_names() and input_params['mongo_params'].drop_collection:
        db.drop_collection(col.name)

    # Para cada año recupera las incidencias utilizando la API de Jira.
    max_year = input_params['jira_api_params'].year
    for month in range(input_params['jira_api_params'].start_month, input_params['jira_api_params'].end_month + 1):
        max_month = month + 1
        if max_month > 12:
            max_month = 1
            max_year += 1
        issues = get_issues_by_date_range(
            project=input_params['jira_api_params'].project,
            min_created_date=f"{input_params['jira_api_params'].year}-{str(month).zfill(2)}-01",
            max_creation_date=f"{max_year}-{str(max_month).zfill(2)}-01",
            max_results=input_params['jira_api_params'].query_limit,
            fields=input_params['jira_api_params'].fields
        )

        save_issues_to_mongodb(mongodb_collection=col, issues=issues)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
