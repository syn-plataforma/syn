import argparse
import math
import os
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient, cursor, DESCENDING, ASCENDING
from pymongo import errors
from pymongo.collection import Collection

from syn.helpers.argparser import mongodb_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger

load_environment_variables()

# Logger.
log = set_logger()


class MongoDBDatasetParams:
    def __init__(
            self,
            db_name="test",
            collection_name="test",
            query_limit=0
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.query_limit = query_limit


class MongoDBFilterParams:
    def __init__(
            self,
            column_name="short_desc",
            start_year=2001,
            end_year=2002
    ):
        self.column_name = column_name
        self.start_year = start_year
        self.end_year = end_year


class MongoDBParams:
    def __init__(
            self,
            host='localhost',
            port=30017,
            db_name="test",
            collection_name="test",
            projection=None,
            query_limit=0,
            drop_collection=False
    ):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.projection = projection
        self.query_limit = query_limit
        self.drop_collection = drop_collection


def get_default_mongo_client() -> MongoClient:
    mongo_api_host = os.environ.get('MONGO_HOST_IP')
    if os.environ.get('ARCHITECTURE') == 'codebooks':
        mongo_api_host = os.environ.get('MONGO_CODEBOOKS_API_HOST')
    log.info(f"Connecting to server: {mongo_api_host}")
    return MongoClient(host=mongo_api_host,
                       port=int(os.environ.get('MONGO_PORT')),
                       username=os.environ.get('MONGODB_USERNAME'),
                       password=os.environ.get('MONGODB_PASSWORD'),
                       authSource=os.environ.get('MONGODB_AUTHENTICATION_DATABASE'),
                       authMechanism='SCRAM-SHA-1')


def get_default_local_mongo_client() -> MongoClient:
    return MongoClient(host='localhost',
                       port=27017)


def save_issues_to_mongodb(mongodb_collection, issues):
    save_to_mongodb_start_time = time.time()

    # Almacena en MongoDB las incidencias obtenidas utilizando la API de MongoDB.
    inserted_docs_number = 0
    if issues is not None and len(issues) > 0:
        insert_many_result = mongodb_collection.insert_many(issues)
        inserted_docs_number = len(insert_many_result.inserted_ids)

    log.info(
        f"Number of inserted documents in "
        f"'{mongodb_collection.name}' = {inserted_docs_number}."
    )
    save_to_mongodb_end_time = time.time()
    log.info(
        f"Total execution time = {((save_to_mongodb_end_time - save_to_mongodb_start_time) / 60)} minutos"
    )


# def save_issues_to_mongodb(mongodb_collection, issues):
#     save_to_mongodb_start_time = time.time()
#
#     # Almacena en MongoDB las incidencias obtenidas utilizando la API de MongoDB.
#     inserted_docs_number = 0
#     if issues is not None and len(issues) > 0:
#         insert_many_result = mongodb_collection.insert_many(issues)
#         inserted_docs_number = len(insert_many_result.inserted_ids)
#
#     save_to_mongodb_end_time = time.time()
#     log.info(
#         f"Tiempo empleado en almacenar los registros en MongoDB = "
#         f"{((save_to_mongodb_end_time - save_to_mongodb_start_time) / 60)} minutos"
#     )
#     log.info(
#         f"Número de incidencias almacenadas en MongoDB en la colección "
#         f"'{mongodb_collection.name}' = {inserted_docs_number}."
#     )


def preprocess_object_mongo(table, table_column_names_array_subcampo):
    for i in range(len(table_column_names_array_subcampo)):
        table[[*table_column_names_array_subcampo[i]][0]] = table[[*table_column_names_array_subcampo[i]][0]].apply(
            lambda x: x[table_column_names_array_subcampo[i][[*table_column_names_array_subcampo[i]][0]]])
    return table


def read_bbdd_tablename(table_name, bbdd_name, mongodb_connection_string, query):
    database = mongodb_connection_string[bbdd_name]
    collection = database[table_name]
    incidencias = collection.find(query)
    table = pd.DataFrame(list(incidencias))
    return table


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[mongodb_parser],
        description='Recupera incidencias de Bugzilla y las almacena en una colección MongoDB.'
    )

    args = parser.parse_args()
    return MongoDBParams(
        host=args.mh,
        port=args.mp,
        db_name=args.db,
        collection_name=args.c,
        projection=args.pj,
        query_limit=args.ql,
        drop_collection=args.dc
    )


def save_dataframe_to_mongodb(database_name: str = 'bugzilla', collection_name: str = 'duplicity_task_train_dataset',
                              dataframe: pd.DataFrame = None, batch_size: int = 10000) -> int:
    log.info(f"Saving DataFrame to MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[database_name]
    col = db[collection_name]

    # Drop collection if exists.
    if col.name in db.list_collection_names():
        log.info(f"Dropping collection: '{db.name}.{col.name}'")
        db.drop_collection(col.name)

    # Save dataset.
    inserted_docs_number = 0
    if dataframe is not None and len(dataframe.index) > 0:
        # Split dataframe in batches to avoid pymongo.errors.CursorNotFound.
        num_batches = math.ceil(dataframe.shape[0] / batch_size)
        batches = np.array_split(dataframe, num_batches)

        for batch in batches:
            insert_many_result = col.insert_many(batch.to_dict("records"))
            inserted_docs_number += len(insert_many_result.inserted_ids)

    log.info(f"Inserted documents into MongoDB: {inserted_docs_number}")
    log.info(f"Saving DataFrame to MongoDB total time: {(time.time() - tic) / 60} minutes")
    return inserted_docs_number


def get_collection(database_name: str = 'bugzilla', collection_name: str = 'normalized_clear') -> Collection:
    # MongoClient connection.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[database_name]
    col = db[collection_name]

    if col.name not in db.list_collection_names():
        raise errors.CollectionInvalid(f"Collection '{database_name}.{collection_name}' not found. "
                                       f"Make sure your collection name is correct.")

    return col


def read_collection_aggregating_and_sorting_by(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query: dict = None,
        projection: dict = None,
        query_limit: int = 0,
        sort_by: str = '',
        ascending: int = -1

) -> cursor:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    col = get_collection(database_name, collection_name)

    # Queries MongoDB collection.
    query = {} if query is None else query
    log.info(f"Query filter document: {query}")
    projection = {'_id': 0} if projection is None else projection
    log.info(f"Projection document: {projection}")
    mongodb_cursor = col.aggregate([
        {'$project': projection},
        {'$match': query},
        {'$sort': {sort_by: ascending}},
        {'$limit': query_limit}
    ],
        allowDiskUse=True
    )
    mongodb_data = list(mongodb_cursor)

    log.info(f"Read documents from MongoDB: {len(mongodb_data)}")
    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return mongodb_data


def read_collection_sorting_by(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query: dict = None,
        projection: dict = None,
        query_limit: int = 0,
        sort_by: str = '',
        order: int = -1

) -> cursor:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    col = get_collection(database_name, collection_name)

    # Queries MongoDB collection.
    query = {} if query is None else query
    log.info(f"Query filter document: {query}")
    projection = {'_id': 0} if projection is None else projection
    log.info(f"Projection document: {projection}")
    # aggregation pipeline
    sort_order = DESCENDING if order == -1 else ASCENDING
    pipeline = [
        {'$match': query},
        {'$project': projection},
        {'$sort': {sort_by: sort_order}},
        {'$limit': query_limit}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    # check MongoDB index
    index = [index['name'] for index in col.list_indexes()]
    if f"{sort_by}_{order}" in index:
        mongodb_cursor = col.aggregate(pipeline)
    else:
        raise NotImplementedError("Sorting MongodDB collection is only available after create an index.")
    mongodb_data = list(mongodb_cursor)

    log.info(f"Read documents from MongoDB: {len(mongodb_data)}")
    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return mongodb_data


def read_collection(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query: dict = None,
        projection: dict = None,
        query_limit: int = 0
) -> cursor:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    col = get_collection(database_name, collection_name)

    # Queries MongoDB collection.
    query = {} if query is None else query
    log.info(f"Query filter document: {query}")
    projection = {'_id': 0} if projection is None else projection
    log.info(f"Projection document: {projection}")
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]

    if query_limit > 0:
        pipeline.append({'$limit': query_limit})

    log.info(f"MongoDB aggregation pipeline. '{pipeline}'")

    mongodb_cursor = col.aggregate(pipeline)
    mongodb_data = list(mongodb_cursor)

    log.info(f"Read documents from MongoDB: {len(mongodb_data)}")
    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return mongodb_data


def load_dataframe_from_mongodb(database_name: str = 'bugzilla', collection_name: str = 'normalized_clear',
                                query: dict = None, projection: dict = None, query_limit: int = 0, sort_by: str = ''
                                ) -> pd.DataFrame:
    # Query the database.
    if sort_by != '':
        mongodb_data = read_collection_sorting_by(
            database_name, collection_name, query, projection, query_limit, sort_by
        )
    else:
        mongodb_data = read_collection(database_name, collection_name, query, projection, query_limit)

    # Expands cursor and builds DataFrame.
    df = pd.DataFrame(list(mongodb_data))

    return df


def read_document(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query: dict = None,
        projection: dict = None
) -> dict:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    col = get_collection(database_name, collection_name)

    # Queries MongoDB collection.
    query = {} if query is None else query
    log.info(f"Query filter document: {query}")
    projection = {'_id': 0} if projection is None else projection
    log.info(f"Projection document: {projection}")
    mongodb_data = col.find_one(query, projection)

    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return mongodb_data


def upsert_document(
        database_name: str = 'tasks',
        collection_name: str = 'experiments',
        query: dict = None,
        document: dict = None
) -> dict:
    log.info(f"Inserting document into: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    col = get_collection(database_name, collection_name)

    # Queries MongoDB collection.
    query = {} if query is None else query
    log.info(f"Query filter document: {query}")

    upserted_document = col.update_one(
        {query},
        {"$set": document},
        upsert=True
    )

    log.info(f"Insert document into MongoDB total time: {(time.time() - tic) / 60} minutes")

    return upserted_document


def find_distinct(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        field: str = ''
) -> dict:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    col = get_collection(database_name, collection_name)

    # Queries MongoDB collection.
    log.info(f"Distinct values for field: '{field}'")
    mongodb_data = col.distinct(field)

    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return mongodb_data
