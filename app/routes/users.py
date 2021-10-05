# -*- coding: utf-8 -*-

import datetime
import os

from flask import Blueprint
from flask import request
from flask import url_for
from flask_jwt_extended import create_access_token

from app.config.config import DevelopmentConfig
from app.models.users import User
from app.utils import responses as resp
from app.utils.responses import response_with
from app.utils.token import generate_verification_token, confirm_verification_token

user_routes = Blueprint("user_routes", __name__)


@user_routes.route('///', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        if User.find_by_email(data['email']) is not None or User.find_by_username(data['username']) is not None:
            return response_with(resp.INVALID_INPUT_422)
        data['password'] = User.generate_hash(data['password'])
        token = generate_verification_token(data['email'])
        verification_email = url_for('user_routes.verify_email', token=token, _external=True)
        result = {'db_insert': str(User.create(data)), 'verification_email': verification_email}
        return response_with(resp.SUCCESS_201, value={'result': result})
    except Exception as e:
        print(e)
        return response_with(resp.INVALID_INPUT_422)


@user_routes.route('/confirm/<token>/', methods=['GET'])
def verify_email(token):
    try:
        email = confirm_verification_token(token)
    except Exception as e:
        return response_with(resp.UNAUTHORIZED_401)
    user = User.find_by_email(email=email)
    if user['is_verified']:
        return response_with(resp.INVALID_INPUT_422)
    else:
        user['is_verified'] = True
        User.update_field(user, 'is_verified', True)
        return response_with(resp.SUCCESS_200, value={'message': 'E-mail verified, you can proceed to login now.'})


@user_routes.route('/dev-login/', methods=['POST'])
def authenticate_dev_user():
    try:
        data = request.get_json()
        current_user = {}
        if data.get('email'):
            current_user = User.find_by_email(data['email'])
        elif data.get('username'):
            current_user = User.find_by_username(data['username'])
        if not current_user:
            return response_with(resp.SERVER_ERROR_404)
        if current_user and not current_user['is_verified']:
            return response_with(resp.BAD_REQUEST_400)
        if User.verify_hash(data['password'], current_user['password']):
            # JWT_ACCESS_TOKEN_EXPIRES en desarrollo el token no expira.
            access_token = create_access_token(identity=current_user['username'], expires_delta=False)
            return response_with(resp.SUCCESS_200,
                                 value={'message': 'Logged in as admin', "access_token": access_token})
        else:
            return response_with(resp.UNAUTHORIZED_401)
    except Exception as e:
        print(e)
        return response_with(resp.INVALID_INPUT_422)


@user_routes.route('/login/', methods=['POST'])
def authenticate_user():
    try:
        data = request.get_json()
        current_user = {}
        if data.get('email'):
            current_user = User.find_by_email(data['email'])
        elif data.get('username'):
            current_user = User.find_by_username(data['username'])
        if not current_user:
            return response_with(resp.SERVER_ERROR_404)
        if current_user and not current_user['is_verified']:
            return response_with(resp.BAD_REQUEST_400)
        if User.verify_hash(data['password'], current_user['password']):
            # JWT_ACCESS_TOKEN_EXPIRES = 15 minutos por defecto
            expires = datetime.timedelta(
                minutes=int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', DevelopmentConfig.JWT_ACCESS_TOKEN_EXPIRES)))
            access_token = create_access_token(identity=current_user['username'], expires_delta=expires)
            return response_with(resp.SUCCESS_200,
                                 value={'message': 'Logged in as admin', "access_token": access_token})
        else:
            return response_with(resp.UNAUTHORIZED_401)
    except Exception as e:
        print(e)
        return response_with(resp.INVALID_INPUT_422)
