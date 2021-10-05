#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
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
    # col = db["clear"]
    #
    # # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    # clear_fields = {
    #     "_id": 0,
    #     "bug_id": 1,
    #     # "product": 1,
    #     "description": 1,
    #     # "bug_severity": 1,
    #     # "dup_id": 1,
    #     # "short_desc": 1,
    #     # "priority": 1,
    #     # "version": 1,
    #     # "component": 1,
    #     # "delta_ts": 1,
    #     "bug_status": 1
    #     # "creation_ts": 1,
    #     # "resolution": 1,
    #     # "short_desc_split_alpha": 1,
    #     # "short_desc_lower": 1,
    #     # "short_desc_punctuaction": 1,
    #     # "short_desc_trim": 1,
    #     # "short_desc_isalpha": 1,
    #     # "short_desc_stop_words": 1,
    #     # "short_desc_diacritic": 1,
    #     # "short_desc_lemmatizer": 1
    #     # "description_split_alpha": 1,
    #     # "description_lower": 1,
    #     # "description_punctuaction": 1,
    #     # "description_trim": 1,
    #     # "description_isalpha": 1,
    #     # "description_stop_words": 1,
    #     # "description_diacritic": 1,
    #     # "description_lemmatizer": 1
    # }

    # # Se utilizarán sólo las incidencias resueltas.
    # query = {
    #     '$or': [
    #         {'bug_status': {'$eq': 'RESOLVED'}},
    #         {'bug_status': {'$eq': 'VERIFIED'}}
    #     ]
    # }
    #
    # data = col.find({}, clear_fields)
    # # .limit(20)
    #
    # # Expande el cursor y construye el DataFrame
    # clear_nlp = pd.DataFrame(list(data))
    #
    # nltk.download("stopwords", quiet=True)
    #
    # clear_nlp["description_split_alpha"] = clear_nlp["description"].apply(lambda x: clean_doc_split(x))
    # clear_nlp["description_lower"] = clear_nlp["description_split_alpha"].apply(lambda x: clean_doc_lower(x))
    #
    # # Elimina la columna 'bug_status' porque se realizará después un merge con la colección original.
    # clear_nlp.drop('bug_status', axis=1, inplace=True)
    # # clear_nlp.drop('description', axis=1, inplace=True)
    #
    # # Almacena el dataframe en MongoDB.
    # db["clear_description_nlp_2"].insert_many(clear_nlp.to_dict('records'))

    fields = {
        "_id": 0
    }

    clear_description_nlp_2_col = db["clear_description_nlp_2"]
    clear_description_nlp_2_data = clear_description_nlp_2_col.find({}, fields)
    clear_description_nlp_2 = pd.DataFrame(list(clear_description_nlp_2_data))

    clear_description_nlp_2["description_punctuaction"] = clear_description_nlp_2["description_lower"].apply(
        lambda x: clean_doc_punctuaction(x))
    clear_description_nlp_2["description_trim"] = clear_description_nlp_2["description_punctuaction"].apply(
        lambda x: clean_doc_trim(x))

    # Almacena el dataframe en MongoDB.
    db["clear_description_nlp_4"].insert_many(clear_description_nlp_2.to_dict('records'))
    #
    # clear_description_nlp_4_col = db["clear_description_nlp_4"]
    # clear_description_nlp_4_data = clear_description_nlp_4_col.find({}, fields)
    # clear_description_nlp_4 = pd.DataFrame(list(clear_description_nlp_4_data))
    # print(clear_description_nlp_4)
    #
    # clear_description_nlp_4["description_isalpha"] = clear_description_nlp_4["description_trim"].apply(
    #     lambda x: clean_doc_isalpha(x))
    # clear_description_nlp_4["description_stop_words"] = clear_description_nlp_4["clear_description_nlp_4"].apply(
    #     lambda x: clean_doc_stopW(x))
    #
    # # Almacena el dataframe en MongoDB.
    # db["clear_description_nlp_6"].insert_many(clear_description_nlp_4.to_dict('records'))
    #
    # clear_description_nlp_6_col = db["clear_description_nlp_6"]
    # clear_description_nlp_6_data = clear_description_nlp_6_col.find({}, fields)
    # clear_description_nlp_6 = pd.DataFrame(list(clear_description_nlp_6_data))
    # print(clear_description_nlp_6)
    #
    # clear_description_nlp_6["description_diacritic"] = clear_description_nlp_6["description_stop_words"].apply(
    #     lambda x: clean_doc_diacri(x))
    # clear_description_nlp_6["description_lemmatizer"] = clear_description_nlp_6["description_diacritic"].apply(
    #     lambda x: clean_doc_lem(x))
    #
    # # Elimina la columna 'description_trim' porque se realizará después un merge con la colección original.
    # clear_description_nlp_6.drop('description_stop_words', axis=1, inplace=True)
    #
    # # Almacena el dataframe en MongoDB.
    # db["clear_description_nlp_8"].insert_many(clear_description_nlp_6.to_dict('records'))
    #
    # # Merge de los dataframes
    # clear_description_nlp_2_4 = pd.merge(clear_description_nlp_2, clear_description_nlp_4, on='bug_id')
    # clear_description_nlp_2_4_6 = pd.merge(clear_description_nlp_2_4, clear_description_nlp_6, on='bug_id')
    #
    # # Almacena el dataframe en MongoDB.
    # db["clear_description_nlp"].insert_many(clear_description_nlp_2_4_6.to_dict('records'))

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
