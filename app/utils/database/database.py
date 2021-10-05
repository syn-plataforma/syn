# -*- coding: utf-8 -*-
import os

from pymongo import MongoClient

from app.config.config import DevelopmentConfig
from syn.helpers.logging import set_logger

log = set_logger()

client = MongoClient(
    host=os.environ.get('MONGO_API_HOST', DevelopmentConfig.MONGO_API_HOST),
    port=int(os.environ.get('MONGO_API_PORT', DevelopmentConfig.MONGO_API_PORT)),
    username=os.environ.get('MONGO_API_USER', DevelopmentConfig.MONGO_API_USER),
    password=os.environ.get('MONGO_API_PASSWORD', DevelopmentConfig.MONGO_API_PASSWORD),
    authSource=os.environ.get('MONGO_API_AUTH_SOURCE', DevelopmentConfig.MONGO_API_AUTH_SOURCE),
    authMechanism=os.environ.get('MONGO_API_AUTH_MECHANISM', DevelopmentConfig.MONGO_API_AUTH_MECHANISM)
)


def get_api_client():
    mongo_api_host = os.environ.get('MONGO_API_HOST', DevelopmentConfig.MONGO_API_HOST)
    if os.environ.get('ARCHITECTURE', DevelopmentConfig.ARCHITECTURE) == 'codebooks':
        mongo_api_host = os.environ.get('MONGO_CODEBOOKS_API_HOST', DevelopmentConfig.MONGO_API_HOST)
    log.info(f"Connecting to server: {mongo_api_host}")
    return MongoClient(
        host=mongo_api_host,
        port=int(os.environ.get('MONGO_API_PORT', DevelopmentConfig.MONGO_API_PORT)),
        username=os.environ.get('MONGO_API_USER', DevelopmentConfig.MONGO_API_USER),
        password=os.environ.get('MONGO_API_PASSWORD', DevelopmentConfig.MONGO_API_PASSWORD),
        authSource=os.environ.get('MONGO_API_AUTH_SOURCE', DevelopmentConfig.MONGO_API_AUTH_SOURCE),
        authMechanism=os.environ.get('MONGO_API_AUTH_MECHANISM', DevelopmentConfig.MONGO_API_AUTH_MECHANISM)
    )


def get_api_database():
    db_name = os.environ.get('MONGO_API_DATABASE', DevelopmentConfig.MONGO_API_DATABASE)
    log.info(f"Connecting to database: {db_name}")
    api_client = get_api_client()
    db = api_client[db_name]
    return db


def get_api_training_parameters_collection():
    collection = os.environ.get('MONGO_API_TRAINING_PARAMETERS_COLLECTION',
                                DevelopmentConfig.MONGO_API_TRAINING_PARAMETERS_COLLECTION)
    log.info(f"Connecting to collection: {collection}")
    training_parameters_collection = get_api_database()[collection]
    return training_parameters_collection


def get_api_user_collection():
    collection = os.environ.get('MONGO_API_USERS_COLLECTION', DevelopmentConfig.MONGO_API_USERS_COLLECTION)
    log.info(f"Connecting to collection: {collection}")
    users_collection = get_api_database()[collection]
    return users_collection


def get_data_client():
    mongo_data_host = os.environ.get('MONGO_DATA_HOST', DevelopmentConfig.MONGO_DATA_HOST)
    if os.environ.get('ARCHITECTURE', DevelopmentConfig.ARCHITECTURE) == 'codebooks':
        mongo_data_host = os.environ.get('MONGO_CODEBOOKS_DATA_HOST', DevelopmentConfig.MONGO_DATA_HOST)
    log.info(f"Connecting to server: {mongo_data_host}")
    return MongoClient(
        host=mongo_data_host,
        port=int(os.environ.get('MONGO_DATA_PORT', DevelopmentConfig.MONGO_DATA_PORT)),
        username=os.environ.get('MONGO_DATA_USER', DevelopmentConfig.MONGO_DATA_USER),
        password=os.environ.get('MONGO_DATA_PASSWORD', DevelopmentConfig.MONGO_DATA_PASSWORD),
        authSource=os.environ.get('MONGO_DATA_AUTH_SOURCE', DevelopmentConfig.MONGO_DATA_AUTH_SOURCE),
        authMechanism=os.environ.get('MONGO_DATA_AUTH_MECHANISM', DevelopmentConfig.MONGO_DATA_AUTH_MECHANISM)
    )


# def get_codebooks_data_client():
#     return MongoClient(
#         host=os.environ.get('MONGO_CODEBOOKS_DATA_HOST', DevelopmentConfig.MONGO_DATA_HOST),
#         port=int(os.environ.get('MONGO_DATA_PORT', DevelopmentConfig.MONGO_DATA_PORT)),
#         username=os.environ.get('MONGO_DATA_USER', DevelopmentConfig.MONGO_DATA_USER),
#         password=os.environ.get('MONGO_DATA_PASSWORD', DevelopmentConfig.MONGO_DATA_PASSWORD),
#         authSource=os.environ.get('MONGO_DATA_AUTH_SOURCE', DevelopmentConfig.MONGO_DATA_AUTH_SOURCE),
#         authMechanism=os.environ.get('MONGO_DATA_AUTH_MECHANISM', DevelopmentConfig.MONGO_DATA_AUTH_MECHANISM)
#     )


def get_data_database(db_name):
    data_client = get_data_client()
    return data_client[db_name]


def mongodb_result_to_dict(query_result):
    output = []

    for result in query_result:
        item = {}
        try:
            for k, v in result.items():
                if k != "_id":
                    item[k] = v
            output.append(item)
        except AttributeError as ae:
            return query_result

    return output
