#!/usr/bin/env python3
import json
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.nlp.TextNormalizer import *

log = set_logger()


def main():
    log.debug(f"\n[INICIO EJECUCIÓN]")
    start_time = time.time()

    # Incializa las variables que almacenarán los argumentos de entrada.
    input_params = get_input_params()

    # Inicializa los parámetros MongoDB para almacenar las estadísticas.
    def get_mongo_client(environment):
        if environment == 'local':
            return MongoClient(host='localhost', port=27017)
        else:
            return get_default_mongo_client()

    # Aplica tqdm a Pandas para poder mostrar barras de progreso en las operaciones.
    tqdm.pandas()

    # Crea el cliente MongoDB.
    mongodb_client: MongoClient = get_mongo_client(input_params['nlp_api_params'].env)
    db = mongodb_client[input_params['mongo_params'].db_name]
    col = db[f"{input_params['mongo_params'].collection_name}"]

    log.debug(f"Normalizando el conjunto de datos:'{db.name}'")

    tic = time.time()
    # Campos seleccionados para los trabajos abordados en el proyecto de investigación.
    fields = {
        "_id": 0,
    }

    # Consulta utilizada para recuperar los datos de MongoDB.
    query = {
        # "bug_id": "296"
    }

    # Recupera la colección de MongoDB.
    clear_data = col.find(query, fields)

    # Expande el cursor y construye el DataFrame
    clear = pd.DataFrame(list(clear_data))
    log.debug(f"[LEER MongoDb: colección '{col.name}'] Tiempo de ejecución : {(time.time() - tic) / 60}")

    # Crea una copia del Dataframe para trabajar.
    aux_clear = clear.copy()

    # Nombre de la colección.
    normalized_col_name = f"normalized_{col.name}"
    log.debug(f"Colecciones exitentes en '{db.name}': {str(db.list_collection_names())}")
    if input_params['nlp_api_params'].split_new_line:
        normalized_col_name = f"{normalized_col_name}_new_line_splited"

    # Colección MongoDB.
    normalized_col = db[normalized_col_name]

    # Si existe una versión previa de la colección MongoDB la elimina.
    if normalized_col.name in db.list_collection_names():
        log.debug(f"Eliminando la colección: '{normalized_col.name}'")
        db.drop_collection(normalized_col.name)

    # Divide el dataframe en subconjuntos para guardar en MongoDB por lotes.
    log.debug(f"Número de lotes: '{input_params['nlp_api_params'].batches}'")
    splitted_df = np.array_split(aux_clear, input_params['nlp_api_params'].batches)

    # Recorre los lotes para normalizar y guardar en MongoDB.
    for df in splitted_df:
        result = None
        # Normaliza las columnas.
        column_names = []
        for column in input_params['nlp_api_params'].columns_names:
            column_names.append(column)
            tic = time.time()
            result = normalize_df_column(df, column, input_params)
            log.debug(f"[NORMALIZAR columna '{column}'] Tiempo de ejecución : {(time.time() - tic) / 60}")

        # Guarda la colección en MongoDB o en un fichero JSON.
        # TODO Crear un Factory para guardar el Dataframe
        tic = time.time()
        if result is not None:
            if input_params['nlp_api_params'].output_format == "mongodb":
                final_result = None
                if input_params['nlp_api_params'].drop_original_columns:
                    final_result = result.drop(columns=column_names, axis=1).copy()
                records = json.loads(final_result.T.to_json()).values()
                normalized_col.insert_many(records)
            if input_params['nlp_api_params'].output_format == "csv":
                result.to_json(f"data/{normalized_col_name}.csv")
            log.debug(
                f"[ESCRIBIR MongoDb: colección '{db.name}_{normalized_col.name}'] "
                f"Tiempo de ejecución : {(time.time() - tic) / 60}")

    log.debug(f"\n[FIN EJECUCIÓN] Tiempo de ejecución : {(time.time() - start_time) / 60}")


if __name__ == '__main__':
    main()
