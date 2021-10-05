#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
import nltk
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.bugzilla.PreprocessClasificattionMongo import *
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
    col = db["clear"]

    # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    fields = {
        "_id": 0,
        "bug_id": 1,
        # "product": 1,
        # "description": 1,
        # "bug_severity": 1,
        # "dup_id": 1,
        "short_desc": 1,
        # "priority": 1,
        # "version": 1,
        # "component": 1,
        # "delta_ts": 1,
        "bug_status": 1
        # "creation_ts": 1,
        # "resolution": 1
    }

    # Se utilizarán sólo las incidencias resueltas.
    query = {
        '$or': [
            {'bug_status': {'$eq': 'RESOLVED'}},
            {'bug_status': {'$eq': 'VERIFIED'}}
        ]
    }

    data = col.find(query, fields)
    # .limit(20)

    # Expande el cursor y construye el DataFrame
    clear_nlp = pd.DataFrame(list(data))

    print(clear_nlp)

    nltk.download("stopwords", quiet=True)

    clear_nlp["short_desc_split_alpha"] = clear_nlp["short_desc"].apply(lambda x: clean_doc_split(x))
    clear_nlp["short_desc_lower"] = clear_nlp["short_desc_split_alpha"].apply(lambda x: clean_doc_lower(x))
    clear_nlp["short_desc_punctuaction"] = clear_nlp["short_desc_lower"].apply(lambda x: clean_doc_punctuaction(x))
    clear_nlp["short_desc_trim"] = clear_nlp["short_desc_punctuaction"].apply(lambda x: clean_doc_trim(x))
    clear_nlp["short_desc_isalpha"] = clear_nlp["short_desc_trim"].apply(lambda x: clean_doc_isalpha(x))
    clear_nlp["short_desc_stop_words"] = clear_nlp["short_desc_isalpha"].apply(lambda x: clean_doc_stopW(x))
    clear_nlp["short_desc_diacritic"] = clear_nlp["short_desc_stop_words"].apply(lambda x: clean_doc_diacri(x))
    clear_nlp["short_desc_lemmatizer"] = clear_nlp["short_desc_diacritic"].apply(lambda x: clean_doc_lem(x))

    # Elimina la columna 'bug_status' porque se realizará después un merge con la colección original.
    clear_nlp.drop('bug_status', axis=1, inplace=True)

    # Almacena el dataframe en MongoDB.
    db["clear_short_desc_nlp"].insert_many(clear_nlp.to_dict('records'))

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
