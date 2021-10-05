import argparse
import json
import os
from pathlib import Path

from definitions import ROOT_DIR
from syn.helpers.argparser import task_parser, source_params_parser, dataset_parser, scheduler_parser, \
    sentence_model_parser, embeddings_parser, structured_data_model_parser, hyper_search_parser, \
    codebooks_parser, classifier_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import read_document, upsert_document

load_environment_variables()
log = set_logger()


def check_params(params: dict = None) -> dict:
    # Check and ensure feasibility
    checked_params = params.copy()

    architecture = checked_params['model']['architecture']
    task = checked_params['dataset']['task']
    corpus = checked_params['dataset']['corpus']
    alias = checked_params['alias']

    # Check params['scheduler']['model_save_dir'].
    if checked_params['scheduler']['model_save_dir'] != '':
        model_save_dir = Path(checked_params['scheduler']['model_save_dir'])
    else:
        model_save_dir = Path(os.environ.get('EXPERIMENTS_PATH')) / architecture / task / corpus / alias

    if not checked_params['reference_params']:
        meta_path = Path(model_save_dir) / 'meta'
        param_path = Path(model_save_dir) / 'param'
        embed_path = Path(model_save_dir) / 'embed'
        model_path = Path(model_save_dir) / 'model'

        if not os.path.exists(meta_path):
            os.makedirs(meta_path)
        if not os.path.exists(param_path) and architecture == 'tree_lstm':
            os.makedirs(param_path)
        if not os.path.exists(embed_path) and architecture == 'tree_lstm':
            os.makedirs(embed_path)
        if not os.path.exists(model_path) and architecture == 'codebooks':
            os.makedirs(model_path)

    checked_params['scheduler']['model_save_dir'] = str(model_save_dir)

    # Check params['dataset']['dataset_save_dir'].
    data_save_dir = Path(os.environ.get('DATA_PATH')) / architecture / task / corpus
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    checked_params['dataset']['dataset_save_dir'] = str(data_save_dir)

    return checked_params


class JsonFileParams:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = JsonFileParams(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path, check_dirs: bool = True):
        with open(json_path) as f:
            params = json.load(f)
            if params is not None:
                if check_dirs:
                    checked_params = check_params(params)
                    self.__dict__.update(checked_params)
                else:
                    self.__dict__.update(params)
            else:
                raise ValueError("No params loaded from JSON file.")

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class MongoDocumentParams:
    """Class that loads hyperparameters from a MongoDB document.
    Example:
    ```
    params = MongoDocumentParams(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(
            self,
            database_name: str = 'tasks',
            collection_name: str = 'experiments',
            params_id: str = ''
    ):
        self.database_name = database_name
        self.collection_name = collection_name
        self.params_id = params_id

        self.query = {'task_id': self.params_id} if self.params_id is not None and '' != self.params_id else None

        self.params = read_document(self.database_name, self.collection_name, self.query)

    def save(self):
        upsert_document(self.database_name, self.collection_name, self.query, self.__dict__)

    def update(self):
        """Loads parameters from MongoDB document"""
        params = read_document(self.database_name, self.collection_name, self.query)
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def get_input_params(check_dirs: bool = True):
    parser = argparse.ArgumentParser(
        parents=[task_parser, source_params_parser, dataset_parser, scheduler_parser, classifier_parser,
                 sentence_model_parser, embeddings_parser, structured_data_model_parser, hyper_search_parser]
    )

    args = parser.parse_args()

    # Params from JSON file.
    # params_file_path = os.path.join(ROOT_DIR, args.params_file)
    params_file_path = Path(ROOT_DIR) / Path(args.params_file)
    if args.params_file is not None and args.params_file != '':
        assert os.path.isfile(params_file_path), f"No json configuration file found at '{params_file_path}'."
        log.info(f"Reading params from filesystem: '{args.params_file}'.")
        return JsonFileParams(params_file_path, check_dirs).dict

    # Params from MongoDB document.
    if args.database_name is not None and args.database_name != '' \
            and args.collection_name is not None and args.collection_name != '' \
            and args.params_id is not None and args.params_id != '':
        log.info(f"Reading params from MongoDB: '{args.database_name}.{args.collection_name}'.")
        return MongoDocumentParams(args.database_name, args.collection_name, args.params_id).dict

    # Params from command line.
    log.info(f"Reading params from command line.")
    params = {
        'alias': args.alias,
        'reference_params': args.reference_params,
        'hyper_search_objective': args.hyper_search_objective,
        'hyper_search_overwrite': args.hyper_search_overwrite,
        'source_params': {
            'params_file': args.params_file,
            'database_name': args.database_name,
            'collection_name': args.collection_name,
            'params_id': args.params_id
        },
        'scheduler': {
            'trainer': args.trainer,
            'sparse': args.sparse,
            'learning_rate_param': args.learning_rate_param,
            'learning_rate_embed': args.learning_rate_embed,
            'learning_rate_decay': 0.99,
            'model_save_dir': args.model_save_dir,
            'batch_size': args.batch_size,
            'loss_function': args.loss_function,
            'regularization_strength': args.regularization_strength,
            'max_epochs': args.max_epochs,
        },
        'model': {
            'architecture': args.architecture,
            'n_classes': args.n_classes,
            'embeddings_size': args.embeddings_size,
            'embeddings_model': args.embeddings_model,
            'embeddings_pretrained': args.embeddings_pretrained,
            'num_layers': args.num_layers,
            'sentence_hidden_dim': args.sentence_hidden_dim,
            'attention': args.attention,
            'attention_dim': args.attention_dim,
            'structured_data_input_dim': args.structured_data_input_dim,
            'structured_data_num_layers': args.structured_data_num_layers,
            'structured_data_hidden_dim': args.structured_data_hidden_dim,
            'structured_data_dropout_rate': args.structured_data_dropout_rate,
            'use_structured_data': args.use_structured_data
        },
        'dataset': {
            'task': args.task,
            'corpus': args.corpus,
            'balance_data': args.balance_data,
            'query_limit': args.query_limit,
        }
    }

    # check and ensure feasibility
    return check_params(params)


def get_codebooks_input_params(check_dirs: bool = True):
    parser = argparse.ArgumentParser(
        parents=[task_parser, source_params_parser, dataset_parser, codebooks_parser, embeddings_parser,
                 hyper_search_parser, classifier_parser]
    )

    args = parser.parse_args()

    # Params from JSON file.
    if args.params_file is not None and args.params_file != '':
        assert os.path.isfile(args.params_file), f"No json configuration file found at '{args.params_file}'."
        log.info(f"Reading params from filesystem: '{args.params_file}'.")
        return JsonFileParams(args.params_file, check_dirs).dict

    # Params from MongoDB document.
    if args.database_name is not None and args.database_name != '' \
            and args.collection_name is not None and args.collection_name != '' \
            and args.params_id is not None and args.params_id != '':
        log.info(f"Reading params from MongoDB: '{args.database_name}.{args.collection_name}'.")
        return MongoDocumentParams(args.database_name, args.collection_name, args.params_id).dict

    # Params from command line.
    log.info(f"Reading params from command line.")
    params = {
        'alias': args.alias,
        'reference_params': args.reference_params,
        'hyper_search_objective': args.hyper_search_objective,
        'hyper_search_overwrite': args.hyper_search_overwrite,
        'source_params': {
            'params_file': args.params_file,
            'database_name': args.database_name,
            'collection_name': args.collection_name,
            'params_id': args.params_id
        },
        'scheduler': {
            'model_save_dir': args.model_save_dir
        },
        'model': {
            'architecture': 'codebooks',
            'classifier': args.classifier,
            'criterion': args.criterion,
            'max_depth': args.max_depth,
            'max_features': args.max_features,
            'random_state': args.random_state,
            'min_samples_leaf': args.min_samples_leaf,
            'penalty': args.penalty,
            'c': args.c,
            'multi_class': args.multi_class,
            'solver': args.solver,
            'gamma': args.gamma,
            'probability': args.probability,
            'n_classes': args.n_classes,
            'embeddings_size': args.embeddings_size,
            'embeddings_model': args.embeddings_model,
            'embeddings_pretrained': args.embeddings_pretrained,
            'codebooks_n_codewords': args.n_codewords,
            'tfidf_min_df': args.min_df,
            'structured_data_input_dim': args.structured_data_input_dim,
            'use_structured_data': args.use_structured_data
        },
        'dataset': {
            'task': args.task,
            'corpus': args.corpus,
            'balance_data': args.balance_data,
            'query_limit': args.query_limit,
        }
    }

    for param in ['criterion', 'max_depth', 'max_features', 'random_state', 'min_samples_leaf', 'penalty', 'c',
                  'multi_class', 'gamma']:
        if args.__dict__[param] is not None:
            params['model'][param.upper() if param == 'd' else param] = args.__dict__[param]

    # check and ensure feasibility
    return check_params(params)


def get_codebooks_trainer_input_params(check_dirs: bool = True):
    parser = argparse.ArgumentParser(
        parents=[task_parser, source_params_parser, dataset_parser, codebooks_parser, embeddings_parser,
                 hyper_search_parser, classifier_parser]
    )

    args = parser.parse_args()

    # Params from JSON file.
    if args.params_file is not None and args.params_file != '':
        assert os.path.isfile(args.params_file), f"No json configuration file found at '{args.params_file}'."
        log.info(f"Reading params from filesystem: '{args.params_file}'.")
        return JsonFileParams(args.params_file, check_dirs).dict

    # Params from MongoDB document.
    if args.database_name is not None and args.database_name != '' \
            and args.collection_name is not None and args.collection_name != '' \
            and args.params_id is not None and args.params_id != '':
        log.info(f"Reading params from MongoDB: '{args.database_name}.{args.collection_name}'.")
        return MongoDocumentParams(args.database_name, args.collection_name, args.params_id).dict

    # Params from command line.
    log.info(f"Reading params from command line.")
    params = {
        'scheduler': {
            'model_save_dir': args.model_save_dir
        },
        'parent_class': args.parent_class,
        'embeddings': {
            'size': args.embeddings_size,
            'model': args.embeddings_model,
            'pretrained': args.embeddings_pretrained
        },
        'codebooks': {
            'n_codewords': args.n_codewords,
            'list_n_codewords': args.list_n_codewords
        },
        'dataset': {
            'task': args.task,
            'corpus': args.corpus,
            'balance_data': args.balance_data,
            'query_limit': args.query_limit,
        }
    }

    # Check and ensure feasibility
    checked_params = params.copy()

    architecture = 'codebooks'
    task = checked_params['dataset']['task']
    corpus = checked_params['dataset']['corpus']

    if checked_params['scheduler']['model_save_dir'] != '':
        model_save_dir = Path(checked_params['scheduler']['model_save_dir'])
    else:
        model_save_dir = Path(os.environ.get('EXPERIMENTS_PATH')) / architecture / task / corpus

    model_path = Path(model_save_dir) / 'codebooks'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checked_params['scheduler']['model_save_dir'] = str(model_save_dir)

    data_save_dir = Path(os.environ.get('DATA_PATH')) / architecture / task / corpus
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    checked_params['dataset']['dataset_save_dir'] = str(data_save_dir)

    return checked_params


def get_codebooks_kwargs_from_params(params):
    return {
        'scheduler': {
            'model_save_dir': str(Path(os.environ.get('EXPERIMENTS_PATH')) / params['model']['architecture'] /
                                  params['dataset']['task'] / params['dataset']['corpus'])
        },
        'parent_class': params['parent_class'],
        'embeddings': {
            'size': params['model']['embeddings_size'],
            'model': params['model']['embeddings_model'],
            'pretrained': params['model']['embeddings_pretrained']
        },
        'codebooks': {
            'n_codewords': params['model']['codebooks_n_codewords']
        },
        'dataset': params['dataset']
    }
