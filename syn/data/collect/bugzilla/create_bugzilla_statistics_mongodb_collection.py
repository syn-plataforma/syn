#!/usr/bin/env python3
import datetime
import json
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.bugzilla.BugzillaHelper import get_bugzilla_bugs_by_date_range
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
    db = mongodb_client[os.environ["BUGZILLA_STATISTICS_MONGODB_DATABASE_NAME"]]
    col = db["bugzilla_statistics"]

    # Inicializa el resultado.
    statistics_dict = {}

    for project in os.environ["BUGZILLA_PROJECT_NAME"].split(","):
        statistics_dict["project"] = project
        total_bugs = 0
        statistics_json = {}
        for year in range(int(os.environ["BUGZILLA_FIRST_CREATION_YEAR"]), datetime.datetime.now().year + 1):
            statistics_dict[year] = {}
            statistics_dict[year]["_total"] = 0
            total_bugs_year = 0
            # month_statistics_dict = {}
            for month in range(1, 13):
                max_year = year
                max_month = month + 1
                if max_month > 12:
                    max_month = 1
                    max_year += 1
                bugs = get_bugzilla_bugs_by_date_range(
                    project=project,
                    min_creation_ts=f"{year}-{str(month).zfill(2)}-01",
                    max_creation_ts=f"{max_year}-{str(max_month).zfill(2)}-01",
                    max_results=0,
                    include_fields="id",
                    get_comments=False
                )
                # Número total de incidencias del mes analizado.
                # month_statistics_dict.append({"month": month, "count": len(bugs)})
                # month_statistics_dict.append({month: len(bugs)})
                statistics_dict[year][month] = len(bugs)
                total_bugs_year += len(bugs)
            # Número total de incidencias del año analizado.
            statistics_dict[year]["_total"] = total_bugs_year
            # Estadísticas mensuales.
            # statistics_dict[year]["bugs"] = month_statistics_dict
            log.info(json.dumps(statistics_dict))
            # Número total de incidencias del proyecto.
            total_bugs += total_bugs_year
        statistics_dict["_total"] = total_bugs
        # Almacena las estadísticas para el proyecto.
        statistics_json = json.dumps(statistics_dict)
        col.insert_one(json.loads(statistics_json))


if __name__ == '__main__':
    main()
