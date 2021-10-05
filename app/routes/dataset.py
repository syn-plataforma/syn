# -*- coding: utf-8 -*-

import os
import time

from flask import Blueprint
from flask_jwt_extended import jwt_required
from pymongo import MongoClient

from app.utils import responses as resp
from app.utils.database.database import get_data_client
from app.utils.responses import response_with
from app.utils.tasks import get_tasks_by_corpus
from syn.helpers.logging import set_logger

log = set_logger()

dataset_routes = Blueprint("dataset_routes", __name__)


# Flask Redirects (301, 302 HTTP responses) from /url to /url/.
# He probado con strict_slashes=False en la ruta y con app.url_map.strict_slashes = False a nivel global, pero no
# resuelve el problema, así que he añadido / al final de todas las rutas.
@dataset_routes.route("/statistics/task/<task>/corpus/<corpus>/", methods=['GET'])
@jwt_required
def get_dataset_statistics_by_task_and_corpus(task, corpus):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[corpus]
    result = {}
    dataset_found = True
    env_list = ['train', 'dev', 'test']
    if os.environ.get('ARCHITECTURE') == 'codebooks':
        env_list = ['train', 'test']

    # custom assignation datasets
    col_name = {
        'train': 'assignation_task_custom_train_dataset_30_lt_2019_11_01',
        'test': 'assignation_task_custom_test_dataset_30_gte_2019_12_01'
    }

    for env in env_list:
        col = db[f"{task}_task_{env}_dataset"]
        if task == 'custom_assignation':
            col = db[col_name[env]]
        if col.name not in db.list_collection_names():
            dataset_found = False
        query = {}
        projection = {}
        data = col.find(query, projection).count()
        result[f"{env}_dataset"] = data

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if dataset_found:
        return response_with(resp.SUCCESS_200, value={"result": [result]})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": []})


@dataset_routes.route("/<corpus>/tasks/", methods=['GET'])
@jwt_required
def get_all_tasks_by_corpus(corpus):
    result = get_tasks_by_corpus(corpus)

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})
