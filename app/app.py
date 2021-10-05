# -*- coding: utf-8 -*-
import logging
import os

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_swagger_ui import get_swaggerui_blueprint

import app.utils.responses as resp
from app.config.config import DevelopmentConfig
from app.routes.dataset import dataset_routes
from app.routes.experiments import experiments_routes
from app.routes.features import features_routes
from app.routes.metrics import metrics_routes
from app.routes.users import user_routes
from app.utils.responses import response_with

app = Flask(__name__)

cors = CORS(app, resources={fr"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/*": {"origins": "*"}})

app.config.from_object(DevelopmentConfig)

app.register_blueprint(experiments_routes,
                       url_prefix=f"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/experiments")

app.register_blueprint(features_routes,
                       url_prefix=f"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/features")

app.register_blueprint(dataset_routes,
                       url_prefix=f"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/dataset")

app.register_blueprint(metrics_routes,
                       url_prefix=f"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/metrics")

app.register_blueprint(user_routes,
                       url_prefix=f"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/users")

# START SWAGGER
SWAGGER_URL = f"/{os.environ.get('API_VERSION', DevelopmentConfig.API_VERSION)}/docs"
API_URL = '/static/swagger.yml'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "SYN REST API"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)


# END SWAGGER

# START GLOBAL HTTP CONFIGURATIONS
@app.after_request
def add_header(response):
    return response


@app.errorhandler(400)
def bad_request(e):
    logging.error(e)
    return response_with(resp.BAD_REQUEST_400)


@app.errorhandler(500)
def server_error(e):
    logging.error(e)
    return response_with(resp.SERVER_ERROR_500)


@app.errorhandler(404)
def not_found(e):
    logging.error(e)
    return response_with(resp.SERVER_ERROR_404)


# END GLOBAL HTTP CONFIGURATIONS


# https://flask-jwt-extended.readthedocs.io/en/stable/
jwt = JWTManager(app)

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0", use_reloader=False)
