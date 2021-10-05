# -*- coding: utf-8 -*-

from flask import Blueprint
from flask_jwt_extended import jwt_required

from app.utils import responses as resp
from app.utils.metrics import get_metrics_by_task
from app.utils.responses import response_with
from syn.helpers.logging import set_logger

log = set_logger()

metrics_routes = Blueprint("metrics_routes", __name__)


# Flask Redirects (301, 302 HTTP responses) from /url to /url/.
# He probado con strict_slashes=False en la ruta y con app.url_map.strict_slashes = False a nivel global, pero no
# resuelve el problema, así que he añadido / al final de todas las rutas.
@metrics_routes.route("/task/<task>/", methods=['GET'])
@jwt_required
def get_all_metrics_by_task(task):
    result = get_metrics_by_task(task)

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})
