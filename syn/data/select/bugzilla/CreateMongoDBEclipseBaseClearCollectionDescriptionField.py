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
    db = mongodb_client["bugzilla"]
    col = db["eclipse_base"]

    # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    fields = {
        "_id": 0,
        # "assigned_to": 1,
        # "assigned_to_detail": 1,
        # "classification": 1,
        # "component": 1,
        # "creation_time": 1,
        # "creator": 1,
        # "creator_detail": 1,
        # "dupe_of": 1,
        "id": 1,
        # "op_sys": 1,
        # "platform": 1,
        # "priority": 1,
        # "product": 1,
        # "resolution": 1,
        # "severity": 1,
        # "status": 1,
        # "summary": 1,
        # "version": 1,
        "description": 1
    }

    data = col.find({}, fields)
    # .limit(20)

    # Expande el cursor y construye el DataFrame
    clear_nlp = pd.DataFrame(list(data))

    print(clear_nlp)

    nltk.download("stopwords", quiet=True)

    clear_nlp["description_split_alpha"] = clear_nlp["description"].astype(str).apply(lambda x: clean_doc_split(x))
    clear_nlp["description_lower"] = clear_nlp["description_split_alpha"].apply(lambda x: clean_doc_lower(x))
    clear_nlp["description_punctuaction"] = clear_nlp["description_lower"].apply(lambda x: clean_doc_punctuaction(x))
    clear_nlp["description_trim"] = clear_nlp["description_punctuaction"].apply(lambda x: clean_doc_trim(x))
    clear_nlp["description_isalpha"] = clear_nlp["description_trim"].apply(lambda x: clean_doc_isalpha(x))
    clear_nlp["description_stop_words"] = clear_nlp["description_isalpha"].apply(lambda x: clean_doc_stopW(x))
    clear_nlp["description_diacritic"] = clear_nlp["description_stop_words"].apply(lambda x: clean_doc_diacri(x))
    clear_nlp["description_lemmatizer"] = clear_nlp["description_diacritic"].apply(lambda x: clean_doc_lem(x))

    # Almacena el dataframe en MongoDB.
    db["eclipse_base_description_clear"].insert_many(clear_nlp.to_dict('records'))

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
