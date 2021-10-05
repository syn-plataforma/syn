# -*- coding: utf-8 -*-

import time

from flask import Blueprint
from flask_jwt_extended import jwt_required
from pymongo import MongoClient

from app.utils import responses as resp
from app.utils.database.database import get_data_client
from app.utils.responses import response_with
from syn.helpers.logging import set_logger
from syn.helpers.task import get_task_request_params_names

log = set_logger()

features_routes = Blueprint("features_routes", __name__)


# Flask Redirects (301, 302 HTTP responses) from /url to /url/.
# He probado con strict_slashes=False en la ruta y con app.url_map.strict_slashes = False a nivel global, pero no
# resuelve el problema, así que he añadido / al final de todas las rutas.
@features_routes.route("/task/<task>/", methods=['GET'])
@jwt_required
def get_features_names(task):
    result = get_task_request_params_names(task)

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@features_routes.route("/<feature>/corpus/<corpus>/", methods=['GET'])
@jwt_required
def get_features_values(feature, corpus):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[corpus]
    col = db[f"{feature}_codes"]
    query = {}
    projection = {'_id': 0, feature: 1}
    data = col.find(query, projection)
    result = []
    for doc in list(data):
        result.append(doc[feature])
    result.sort()

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})
