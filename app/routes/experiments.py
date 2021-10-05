# -*- coding: utf-8 -*-

import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from pymongo import MongoClient
from tqdm import tqdm

from app.config.config import DevelopmentConfig
from app.utils import responses as resp
from app.utils.database.database import get_data_client
from app.utils.dataset import get_pairs_dataset, format_dataset, format_codebooks_dataset
from app.utils.responses import response_with
from app.utils.tasks import get_task_type_text
from syn.data.clean.NormalizeText import normalize_incidence, get_codebooks_tokens
from syn.helpers.logging import set_logger
from syn.helpers.nlp.embeddings import get_embeddings, get_filtered_word_embeddings_filename
from syn.helpers.nlp.stanford import get_collapsed_unary_binary_trees
from syn.helpers.nlp.trees import get_attention_vectors
from syn.helpers.task import get_task_features_column_names, get_task_request_params_names, get_label_column_name
from syn.helpers.treelstm.dataset import encode_dataset_structured_data, get_task_structured_data_codes
from syn.model.build.codebooks.codebooks_model import get_codebooks_model
from syn.model.build.common.task import GridSearch
from syn.model.build.treelstm.dynetconfig import get_dynet
from syn.model.build.treelstm.dynetmodel import get_dynet_model
from syn.model.build.treelstm.dynettask import check_data_integrity

log = set_logger()
dy = get_dynet()

experiments_routes = Blueprint("experiments_routes", __name__)


# Flask Redirects (301, 302 HTTP responses) from /url to /url/.
# He probado con strict_slashes=False en la ruta y con app.url_map.strict_slashes = False a nivel global, pero no
# resuelve el problema, así que he añadido / al final de todas las rutas.
@experiments_routes.route("///", methods=['GET'])
@jwt_required
def get_all_experiments():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {}
    projection = {
        '_id': 0,
        'task': '$task_action.kwargs.dataset.task',
        'corpus': '$task_action.kwargs.dataset.corpus',
        'classifier': '$task_action.kwargs.model.classifier',
        'description': '$task_name',
        'task_id': 1
    }

    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result = list(data)

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/<experiment_id>/info/", methods=['GET'])
@jwt_required
def get_experiment_info(experiment_id):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_id': str(experiment_id)}
    projection = {'_id': 0}
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result = list(data)

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/tasks/", methods=['GET'])
@jwt_required
def get_distinct_experiments_by_task():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    data = col.distinct('task_action.kwargs.dataset.task')
    result = []
    for task in data:
        result.append({'task_name': get_task_type_text(task, 'es'), 'task': task})

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/task/<task>/", methods=['GET'])
@jwt_required
def get_all_experiments_by_task(task):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_action.kwargs.dataset.task': str(task)}
    projection = {
        '_id': 0,
        'task': '$task_action.kwargs.dataset.task',
        'corpus': '$task_action.kwargs.dataset.corpus',
        'classifier': '$task_action.kwargs.model.classifier',
        'description': '$task_name',
        'task_id': 1
    }

    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)

    result = list(data)

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/task/<task>/corpus/", methods=['GET'])
@jwt_required
def get_distinct_corpus_in_experiments_by_task(task):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_action.kwargs.dataset.task': str(task)}
    projection = {'_id': 0, 'corpus': '$task_action.kwargs.dataset.corpus'}
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result_set = set()
    for doc in data:
        result_set.add(doc['corpus'])

    result = list(result_set)
    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/task/<task>/corpus/<corpus>/", methods=['GET'])
@jwt_required
def get_all_experiment_by_task_and_corpus(task, corpus):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_action.kwargs.dataset.task': str(task), 'task_action.kwargs.dataset.corpus': str(corpus)}
    projection = {
        '_id': 0,
        'task': '$task_action.kwargs.dataset.task',
        'corpus': '$task_action.kwargs.dataset.corpus',
        'classifier': '$task_action.kwargs.model.classifier',
        'description': '$task_name',
        'task_id': 1
    }
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result = list(data)

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/<experiment_id>/model/hyperparameters/", methods=['GET'])
@jwt_required
def get_experiment_hyperparameters(experiment_id):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_id': str(experiment_id), 'task_action.kwargs': {'$exists': True}}
    projection = {'_id': 0, 'hyperparameters': '$task_action.kwargs'}
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result = list(data)

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/<experiment_id>/model/train-info/", methods=['GET'])
@jwt_required
def get_experiment_train_info(experiment_id):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_id': str(experiment_id), 'task_action.train': {'$exists': True}}
    projection = {'_id': 0, 'train': '$task_action.train'}
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result = list(data)

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": list(data)})


@experiments_routes.route("/<experiment_id>/model/metrics/", methods=['GET'])
@jwt_required
def get_experiment_metrics(experiment_id):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # MongoDB data.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {'task_id': str(experiment_id), 'task_action.evaluation.metrics': {'$exists': True}}
    projection = {'_id': 0, 'metrics': '$task_action.evaluation.metrics'}
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregation pipeline: '{pipeline}'")
    data = col.aggregate(pipeline)
    result = list(data)
    print(len(result))

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


# @experiments_routes.route("/<experiment_id>/model/predict/", methods=['POST'])
# @jwt_required
# def predict(experiment_id):
#     # Stores the execution start time to calculate the time it takes for the module to execute.
#     initial_time = time.time()
#
#     # Load saved task.
#     mongodb_client: MongoClient = get_data_client()
#     db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
#     col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
#     query = {
#         'task_id': str(experiment_id),
#         'task_action.kwargs': {'$exists': True},
#         'task_action.train': {'$exists': True}
#     }
#     projection = {
#         '_id': 0,
#         'kwargs': '$task_action.kwargs',
#         'model_meta_file': '$task_action.train.model_meta_file'
#     }
#     # aggregation pipeline
#     pipeline = [
#         {'$match': query},
#         {'$project': projection}
#     ]
#     log.info(f"Aggregate pipeline: {pipeline}")
#     data = col.aggregate(pipeline)
#     kwargs = data['kwargs']
#     model_meta_file = data['model_meta_file']
#
#     saved_params = {}
#     if model_meta_file is not None:
#         # Change path to Docker container.
#         if os.name == 'posix':
#             model_meta_file = model_meta_file.replace('/datadrive/host-mounted-volumes/syn/', '/usr/src/')
#         log.info(f"Loading hyperparameters from: '{str(Path(model_meta_file))}'.")
#         try:
#             saved_params = np.load(str(Path(model_meta_file)), allow_pickle=True).item()
#         except FileNotFoundError:
#             return response_with(resp.SERVER_ERROR_404, value={"result": []})
#
#     # Get collapsed unary binary trees
#     collapsed_unary_binary_trees = get_collapsed_unary_binary_trees(
#         normalize_incidence(str(request.json['description']), to_lower_case=True)
#     )
#
#     if len(collapsed_unary_binary_trees) == 0:
#         return response_with(resp.INVALID_INPUT_422, value={"result": []})
#
#     # Get attention vectors
#     attention_vectors = get_attention_vectors(collapsed_unary_binary_trees)
#
#     # Data
#     data_columns = get_task_features_column_names(kwargs['dataset']['task'])
#     dataset = pd.DataFrame(columns=data_columns)
#
#     request_args = [i for i in request.json.keys()]
#     for column in get_task_request_params_names(kwargs['dataset']['task']):
#         if column in request_args:
#             dataset.at[0, column] = request.json[column]
#
#     dataset.at[0, 'trees'] = collapsed_unary_binary_trees
#     dataset.at[0, 'attention_vectors'] = attention_vectors
#
#     # if task is 'duplicity' or 'similarity' need do more actions.
#     query_limit = 1000
#     if 'num_issues_to_compare' in request.args.keys() and int(request.args['num_issues_to_compare']) > 0:
#         query_limit = int(request.args['num_issues_to_compare'])
#     if kwargs['dataset']['task'] in ['duplicity', 'similarity']:
#         dataset = get_pairs_dataset(
#             arch='codebooks',
#             dataset=dataset,
#             task=kwargs['dataset']['task'],
#             corpus=kwargs['dataset']['corpus'],
#             query_limit=query_limit
#         ).copy()
#
#     # Encode structured info.
#     dataset = encode_dataset_structured_data(dataset, kwargs['dataset']['corpus'], kwargs['dataset']['task'])
#
#     # Format dataset
#     inst = format_dataset(dataset, kwargs['dataset']['task'], kwargs['dataset']['corpus'])
#
#     # Word embeddings.
#     embeddings_dir = Path(os.environ.get('DATA_PATH', DevelopmentConfig.DATA_PATH)) / 'word_embeddings'
#     if saved_params['embeddings_pretrained'] or 'glove' == saved_params['embeddings_model']:
#         embeddings_filename = \
#             get_filtered_word_embeddings_filename(
#                 model=saved_params['embeddings_model'],
#                 size=saved_params['embeddings_size']
#             )
#     else:
#         embeddings_filename = f"{saved_params['embeddings_model']}-{kwargs['dataset']['corpus']}-" \
#                               f"{saved_params['embeddings_size']}.txt"
#
#     embeddings_path = Path(embeddings_dir) / saved_params['embeddings_model'] / embeddings_filename
#     # Change path to Docker container.
#     if os.name == 'posix':
#         embeddings_path = str(embeddings_path).replace('/datadrive/host-mounted-volumes/syn/data/', '/usr/src/')
#     if not os.path.isfile(embeddings_path):
#         log.error(f"No such filtered word embeddings file: '{embeddings_path}'.")
#     assert os.path.isfile(embeddings_path), 'Ensure word embeddings file exists.'
#     log.info(f"Reading embeddings from '{embeddings_path}' ...")
#     word_embed, w2i = get_embeddings(embed_path=embeddings_path, embeddings_size=saved_params['embeddings_size'])
#
#     # Load model
#     model_builder = get_dynet_model(kwargs['dataset']['task'])
#
#     try:
#         model = model_builder(
#             n_classes=saved_params['n_classes'],
#             w2i=w2i,
#             word_embed=word_embed,
#             params=saved_params,
#             model_meta_file=model_meta_file
#         )
#     except FileNotFoundError:
#         return response_with(resp.SERVER_ERROR_404, value={"result": []})
#
#     # build graph for this instance
#     dy.renew_cg()
#     result = []
#     df_result = pd.DataFrame(columns=['bug_id', 'predidct_proba'])
#     if kwargs['dataset']['task'] not in ['duplicity', 'similarity']:
#         # Check data integrity.
#         check_data_integrity(inst[0], inst[1])
#
#         # Issue description as Tuple(trees, attention_vectors).
#         issue_description = (inst[0], inst[1])
#
#         # Issue structured data.
#         issue_structured_data = inst[2]
#
#         pred, predict_proba, _ = model.predict(issue_description, issue_structured_data)
#
#         label = get_label_column_name(kwargs['dataset']['task'], kwargs['dataset']['corpus'])
#         codes = get_task_structured_data_codes(kwargs['dataset']['corpus'], label)
#
#         result.append({'pred': list(codes.keys())[list(codes.values()).index(pred)]})
#     else:
#         for i, pair in enumerate(tqdm(inst, total=len(inst), desc='rows')):
#             # Tuple(inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6]) =
#             # Tuple(trees_left, trees_right, attention_vectors_left, attention_vectors_right, structured_data_left,
#             # structured_data_right, label).
#
#             # build graph for this instance
#             dy.renew_cg()
#
#             # Check data integrity.
#             check_data_integrity(pair[0], pair[2])
#             check_data_integrity(pair[1], pair[3])
#
#             # Issue description as Tuple(trees, attention_vectors).
#             issue_description_left = (pair[0], pair[2])
#             issue_description_right = (pair[1], pair[3])
#
#             # Issue structured data.
#             issue_structured_data_left = pair[4]
#             issue_structured_data_right = pair[5]
#
#             pred, predict_proba, _, _ = model.predict(issue_description_left, issue_description_right,
#                                                       issue_structured_data_left, issue_structured_data_right)
#
#             if pred == 1:
#                 df_result.at[i, 'bug_id'] = pair[6]
#                 df_result.at[i, 'predidct_proba'] = predict_proba[pred]
#
#         max_num_predictions = 5
#         if 'max_num_predictions' in request.args.keys() and int(request.args['max_num_predictions']) > 0:
#             max_num_predictions = int(request.args['max_num_predictions'])
#         sorted_df_result = df_result.sort_values('predidct_proba', ascending=False).copy()
#         limited_sorted_df_result = sorted_df_result.head(max_num_predictions).copy()
#         for index, row in limited_sorted_df_result.iterrows():
#             result.append({'bug_id': int(row['bug_id']), 'predidct_proba': float(row['predidct_proba'])})
#         log.info(f"result = {result}")
#
#     final_time = time.time()
#     log.info(f"Total execution time = {final_time - initial_time} seconds")
#
#     return response_with(resp.SUCCESS_200, value={"result": result})


# @experiments_routes.route("/<experiment_id>/codebooks/model/predict/", methods=['POST'])
@experiments_routes.route("/<experiment_id>/model/predict/", methods=['POST'])
@jwt_required
def codebooks_predict(experiment_id):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Load saved task.
    mongodb_client: MongoClient = get_data_client()
    db = mongodb_client[os.environ.get('TASKS_DATABASE_NAME', DevelopmentConfig.TASKS_DATABASE_NAME)]
    col = db[os.environ.get('EXPERIMENTS_COLLECTION_NAME', DevelopmentConfig.EXPERIMENTS_COLLECTION_NAME)]
    query = {
        'task_id': str(experiment_id),
        'task_action.kwargs': {'$exists': True},
        'task_action.train': {'$exists': True}
    }
    projection = {
        '_id': 0,
        'kwargs': '$task_action.kwargs',
        'model_meta_file': '$task_action.train.model_meta_file'
    }
    # aggregation pipeline
    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]
    log.info(f"Aggregate pipeline: {pipeline}")
    data = list(col.aggregate(pipeline))
    if len(data) == 0:
        return response_with(resp.SERVER_ERROR_404, value={"result": []})
    kwargs = data[0]['kwargs']
    model_meta_file = data[0]['model_meta_file']

    saved_params = {}
    if model_meta_file is not None:
        # Change path to Docker container.
        if os.name == 'posix':
            model_meta_file = model_meta_file.replace('/datadrive/host-mounted-volumes/syn/', '/usr/src/')
        log.info(f"Loading hyperparameters from: '{str(Path(model_meta_file))}'.")
        try:
            saved_params = np.load(str(Path(model_meta_file)), allow_pickle=True).item()
        except FileNotFoundError:
            return response_with(resp.SERVER_ERROR_404, value={"result": []})

    # Data
    data_columns = get_task_features_column_names(kwargs['dataset']['task'], os.environ.get('ARCHITECTURE'))
    dataset = pd.DataFrame(columns=data_columns)

    request_args = [i for i in request.json.keys()]
    for column in get_task_request_params_names(kwargs['dataset']['task']):
        if column in request_args:
            dataset.at[0, column] = request.json[column]

    if kwargs['dataset']['task'] == 'custom_assignation':
        # get composite_data
        composite_data = []
        for field in os.environ["COMPOSITE_ID_FIELDS"].split(","):
            composite_data.append(dataset.at[0, field])

        dataset.at[0, 'composite_data'] = composite_data

    # tokens
    tokens = get_codebooks_tokens(normalize_incidence(str(request.json['description']), to_lower_case=True))
    dataset.at[0, 'tokens'] = tokens
    # if task is 'duplicity' or 'similarity' need do more actions.
    query_limit = 1000
    if 'num_issues_to_compare' in request.args.keys() and int(request.args['num_issues_to_compare']) > 0:
        query_limit = int(request.args['num_issues_to_compare'])
    if kwargs['dataset']['task'] in ['duplicity', 'similarity']:
        dataset = get_pairs_dataset(
            arch='codebooks',
            dataset=dataset,
            task=kwargs['dataset']['task'],
            corpus=kwargs['dataset']['corpus'],
            query_limit=query_limit
        ).copy()

    # Encode structured info.
    dataset = encode_dataset_structured_data(dataset, kwargs['dataset']['corpus'], kwargs['dataset']['task'])

    # Format dataset
    inst = format_codebooks_dataset(dataset, kwargs['dataset']['task'], kwargs['dataset']['corpus'])

    # Word embeddings
    word_embed = None
    w2i = None
    if kwargs['dataset']['task'] != 'custom_assignation':
        embeddings_dir = Path(os.environ.get('DATA_PATH', DevelopmentConfig.DATA_PATH)) / 'word_embeddings'
        if saved_params['model']['embeddings_pretrained'] or 'glove' == saved_params['model']['embeddings_model']:
            embeddings_filename = \
                get_filtered_word_embeddings_filename(
                    corpus=saved_params['dataset']['corpus'],
                    model=saved_params['model']['embeddings_model'],
                    size=saved_params['model']['embeddings_size']
                )
        else:
            embeddings_filename = f"{saved_params['embeddings_model']}-{kwargs['dataset']['corpus']}-" \
                                  f"{saved_params['embeddings_size']}.txt"

        embeddings_path = Path(embeddings_dir) / saved_params['model']['embeddings_model'] / embeddings_filename
        # Change path to Docker container.
        if os.name == 'posix':
            embeddings_path = str(embeddings_path).replace('/datadrive/host-mounted-volumes/syn/data/', '/usr/src/')
        if not os.path.isfile(embeddings_path):
            log.error(f"No such filtered word embeddings file: '{embeddings_path}'.")
        assert os.path.isfile(embeddings_path), 'Ensure word embeddings file exists.'
        log.info(f"Reading embeddings from '{embeddings_path}' ...")
        word_embed, w2i = get_embeddings(
            embed_path=embeddings_path, embeddings_size=saved_params['model']['embeddings_size']
        )

    # Load model
    model_builder = get_codebooks_model(kwargs['dataset']['task'])
    try:
        model = model_builder(
            params=saved_params,
            model_meta_file=model_meta_file
        )
    except FileNotFoundError:
        return response_with(resp.SERVER_ERROR_404, value={"result": []})
    log.info(f"Model loaded: '{model}'")

    # initialize response
    result = []
    df_result = pd.DataFrame(columns=['bug_id', 'predidct_proba'])

    # custom assignation
    if kwargs['dataset']['task'] == 'custom_assignation':
        prediction, _ = model.predict(dataset)

        for pred in prediction[0]:
            result.append({'pred': pred['label']})
    else:
        log.info(f"Calculating predictions ....")
        # other tasks
        if kwargs['dataset']['task'] not in ['duplicity', 'similarity']:
            # Tuple(inst[0], inst[1]) = Tuple(tokens, structured_data).
            # issue description
            issue_description = inst[0]

            # issue structured data
            issue_structured_data = [item[0] for item in inst[1]]

            # TF-IDF features
            issue_description_repr = model.issue_description_builder.transform(
                issue_description, w2i, word_embed
            )

            issue_description_repr_dense = []
            num_rows = issue_description_repr.shape[0]
            for row in tqdm(issue_description_repr, total=num_rows, desc='rows'):
                issue_description_repr_dense += row.todense().tolist()

            # issue representation.
            if model.params['model']['use_structured_data']:
                issue_repr = []
                left_array = issue_description_repr_dense[0]
                right_array = issue_structured_data
                left_array.extend(right_array)
                issue_repr.append(left_array)
            else:
                issue_repr = list(issue_description_repr_dense)

            # classifier prediction
            prediction = model.model.predict(list(issue_repr))

            label = get_label_column_name(kwargs['dataset']['task'], kwargs['dataset']['corpus'])
            codes = get_task_structured_data_codes(kwargs['dataset']['corpus'], label)

            result.append({'pred': list(codes.keys())[list(codes.values()).index(prediction)]})
        else:
            log.info(f"Calculating predictions ....")
            for i, pair in enumerate(tqdm(inst, total=len(inst), desc='rows')):
                # Tuple(inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6]) =
                # Tuple(tokens_left, tokens_right, structured_data_left, structured_data_right).

                # Issue description
                issue_description_left = pair[0]
                issue_description_right = pair[1]

                # Issue structured data.
                issue_structured_data_left_repr = [item[0] for item in pair[2]]
                issue_structured_data_right_repr = [item[0] for item in pair[3]]

                # TF-IDF features
                # use left and right issues descriptions
                left_right_issuers_descriptions = issue_description_left + issue_description_right
                issue_description_total_repr = model.issue_description_builder.transform(
                    left_right_issuers_descriptions,
                    w2i,
                    word_embed
                )

                # Transform scipy.sparse.csr.csr_matrix to dense array
                issue_description_total_repr_dense = []
                num_rows = issue_description_total_repr.shape[0]
                for row in tqdm(issue_description_total_repr, total=num_rows, desc='rows'):
                    issue_description_total_repr_dense += row.todense().tolist()

                # split into left and right issues representations
                issue_description_left_repr = issue_description_total_repr_dense[:len(issue_description_left)]
                issue_description_right_repr = issue_description_total_repr_dense[len(issue_description_right):]

                # issue representation.
                if model.params['model']['use_structured_data']:
                    issue_left_repr = []
                    issue_right_repr = []

                    left_issue_description_left_array = issue_description_left_repr[0]
                    right_issue_structured_data_left_array = issue_structured_data_left_repr
                    left_issue_description_left_array.extend(right_issue_structured_data_left_array)
                    issue_left_repr.append(left_issue_description_left_array)

                    left_issue_description_right_array = issue_description_right_repr[0]
                    right_issue_structured_data_right_array = issue_structured_data_right_repr
                    left_issue_description_right_array.extend(right_issue_structured_data_right_array)
                    issue_right_repr.append(left_issue_description_right_array)

                else:
                    issue_left_repr = list(issue_description_left_repr)
                    issue_right_repr = list(issue_description_right_repr)

                # Left and right issues representations, and similarity representation.
                concatenated_data = []
                subtract_row = [a_i - b_i for a_i, b_i in zip(issue_left_repr[0], issue_right_repr[0])]
                abs_row = [abs(ele) for ele in subtract_row]
                multiply_row = [a_i * b_i for a_i, b_i in zip(issue_left_repr[0], issue_right_repr[0])]
                abs_row.extend(multiply_row)
                abs_row.extend(issue_left_repr[0])
                abs_row.extend(issue_right_repr[0])
                concatenated_data.append(abs_row)

                # classifier prediction
                prediction = model.model.predict(list(concatenated_data))
                prediction_probability = model.model.predict_proba(list(concatenated_data))

                if prediction == 1:
                    df_result.at[i, 'bug_id'] = pair[4]
                    df_result.at[i, 'predidct_proba'] = prediction_probability[0][prediction]

            max_num_predictions = 5
            if 'max_num_predictions' in request.args.keys() and int(request.args['max_num_predictions']) > 0:
                max_num_predictions = int(request.args['max_num_predictions'])
            sorted_df_result = df_result.sort_values('predidct_proba', ascending=False).copy()
            limited_sorted_df_result = sorted_df_result.head(max_num_predictions).copy()
            for index, row in limited_sorted_df_result.iterrows():
                result.append({'bug_id': int(row['bug_id']), 'predidct_proba': float(row['predidct_proba'])})
            log.info(f"result = {result}")

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    return response_with(resp.SUCCESS_200, value={"result": result})


@experiments_routes.route("/task/<task>/corpus/<corpus>/best-model/", methods=['GET'])
@jwt_required
def get_best_model_by_task_and_corpus(task, corpus):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    parameter_space = {
        'dataset': {
            'task': task,
            'corpus': corpus
        }
    }

    request_args = [i for i in request.args.keys()] if request.args is not None else []
    objective = request.args['objective'].split(',') if 'objective' in request_args else ['accuracy']
    tuner = GridSearch(objective=objective, parameter_space=parameter_space)
    try:
        models = tuner.get_best_models(num_models=1)
    except (ValueError, AssertionError):
        return response_with(resp.SERVER_ERROR_404, value={"result": []})

    result = [{'task_id': models[0]['task_id']}]

    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})


@experiments_routes.route("/task/<task>/corpus/<corpus>/aggregated-metrics/", methods=['GET'])
@jwt_required
def get_aggregated_metrics_by_task_and_corpus(task, corpus):
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    parameter_space = {
        'dataset': {
            'task': task,
            'corpus': corpus
        }
    }

    request_args = [i for i in request.args.keys()] if request.args is not None else []
    objective = request.args['objective'].split(',') if 'objective' in request_args else ['accuracy']
    tuner = GridSearch(objective=objective, parameter_space=parameter_space)

    try:
        metrics = tuner.aggregate_metrics()
    except (ValueError, AssertionError):
        return response_with(resp.SERVER_ERROR_404, value={"result": []})

    # change NaN for None
    for metric in metrics:
        for k in metrics[metric].keys():
            if isinstance(metrics[metric][k], float) and math.isnan(metrics[metric][k]):
                metrics[metric][k] = None

    excluded_columns = ['task', 'corpus', 'balance_data', 'architecture', 'dataset_save_dir', 'model_save_dir']
    for key in metrics:
        for column in excluded_columns:
            metrics[key].pop(column, None)

    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[idx] + [values[h] for h in headers] for idx, values in metrics.items()]
    result = {'headers': list(headers), 'table': table}
    final_time = time.time()
    log.info(f"Total execution time = {final_time - initial_time} seconds")

    if len(result) > 0:
        return response_with(resp.SUCCESS_200, value={"result": result})
    else:
        return response_with(resp.SERVER_ERROR_404, value={"result": result})
