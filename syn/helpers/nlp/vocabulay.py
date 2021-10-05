import time
from typing import Union

import pandas as pd
from pymongo import MongoClient, errors

from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client, save_dataframe_to_mongodb, load_dataframe_from_mongodb

# Logger.
log = set_logger()


def get_tokens_query() -> dict:
    return {
        'detailed_tokens.0.0': {
            '$exists': 'true'
        }
    }


def get_tokens_projection() -> dict:
    return {
        "_id": 0,
        "tokens": "$detailed_tokens"
    }


def load_tokens_from_mongodb(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query_limit: int = 0
) -> pd.DataFrame:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[database_name]
    col = db[collection_name]

    if col.name not in db.list_collection_names():
        raise errors.CollectionInvalid(f"Collection '{db.name}.{col.name}' not found. "
                                       f"Make sure your collection name is correct.")

    # Queries MongoDB collection.
    query = get_tokens_query()
    log.info(f"Query filter document: {query}")
    projection = get_tokens_projection()
    log.info(f"Projection document: {projection}")
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]

    if query_limit > 0:
        pipeline.append({'$limit': query_limit})

    mongodb_data = col.aggregate(pipeline)
    # Expands cursor and builds DataFrame.
    df = pd.DataFrame(list(mongodb_data))
    log.info(f"Read documents from MongoDB: {len(df.index)}")
    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return df


def load_tokens(
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query_limit: int = 0
) -> Union[None, pd.DataFrame]:
    return load_tokens_from_mongodb(database_name=database_name, collection_name=collection_name,
                                    query_limit=query_limit)


def get_vocabulary_from_mongodb(row: list) -> set:
    vocab = set()
    for sentence in row:
        for token in sentence:
            vocab.add(token)
    return vocab


def save_vocabulary_to_mongodb(database_name: str = 'bugzilla', collection_name: str = 'duplicity_task_vocabulary',
                               vocabulary: set = None) -> int:
    words = [{'word': word} for word in vocabulary]
    # Returns save vocabulary function.
    return save_dataframe_to_mongodb(database_name=database_name, collection_name=collection_name,
                                     dataframe=pd.DataFrame(words))


def save_vocabulary(
        database_name: str = 'bugzilla',
        collection_name: str = 'vocabulary',
        vocabulary: set = None
) -> Union[None, int]:
    return save_vocabulary_to_mongodb(database_name=database_name, collection_name=collection_name,
                                      vocabulary=vocabulary)


def load_vocabulary_from_mongodb(database_name: str = 'bugzilla', collection_name: str = 'duplicity_task_vocabulary',
                                 query: dict = None, projection: dict = None, query_limit: int = 0) -> set:
    # Read MongoDB collection.
    df = load_dataframe_from_mongodb(database_name=database_name, collection_name=collection_name, query=query,
                                     projection=projection, query_limit=query_limit)
    return set(df['word'].values.tolist())


def load_vocabulary(
        database_name: str = 'bugzilla',
        collection_name: str = 'vocabulary',
        query_limit: int = 0
) -> Union[None, set]:
    return load_vocabulary_from_mongodb(database_name=database_name, collection_name=collection_name, query=None,
                                        projection=None, query_limit=query_limit)
