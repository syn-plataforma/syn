"""Tasks module"""
import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Any

import pandas as pd
from pymongo import MongoClient
from tabulate import tabulate
from tqdm import tqdm

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client, load_dataframe_from_mongodb, \
    read_collection
from syn.model.build.common import consensus

load_environment_variables()
log = set_logger()


def _hash_function(w):
    """Hash function to provide a unique (negligible collision) string identifier from a dict of parameters"""
    h = hashlib.md5(w)
    return base64.b64encode(h.digest())[:12].decode("utf-8").replace("/", "_")


def _identity(x):
    """An identity function"""
    return x


def preprocess_list(input_list):
    new_list = []
    for item in input_list:
        new_list.append([item])
    return new_list


def postprocess_array(arr):
    result = []
    for it in arr:
        result.append(it)
    return result


def get_task_kwargs(params):
    return {
        'scheduler': params['scheduler'],
        'model': params['model'],
        'dataset': params['dataset']
    }


class Task(ABC):
    """A Task able to provide a result from a dict of parameters"""

    def __init__(self):
        self.kwargs = {}
        self.task_id = None
        self.task_name = ''
        self.task_action = {}
        self.save_dbname = 'syn'
        self.save_collection = 'tasks'

    def get_hash(self):
        """Get a hash identifying the task"""
        str_form = json.dumps(self.kwargs, sort_keys=True)
        return _hash_function(str_form.encode('utf-8'))

    def get_file_name(self):
        """Get a unique filename for the task"""
        return "%s-%s" % (self.__class__.__name__, self.get_hash())

    @abstractmethod
    def load(self):
        """Load saved task"""
        raise NotImplementedError("This method must be inherited")

    @abstractmethod
    def run(self):
        """Execute the task"""
        raise NotImplementedError("This method must be inherited")

    def delete(self):
        os.remove(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name())

    def get_model_dump_info(self, class_name=None):
        return {
            '_id': self.get_hash(),
            '_type': class_name if class_name is not None else self.__class__.__name__
        }

    def run_and_save(self):
        """Execute the task and store the results as a pickle and in the db"""
        # Run and save in filesystem.
        result = self.run()

        # Save to MongoDB.
        self._db_store()

        return result

    def load_or_run(self):
        """Load the results if available, otherwise running the task, storing the results, and returning them"""
        try:
            return self.load()
        except FileNotFoundError:
            return self.run_and_save()

    def _db_store(self):
        """Store the task in the db"""
        log.info(f"Storing data in MongoDB ...")
        initial_time = time.time()

        client: MongoClient = get_default_mongo_client()

        query = {'task_id': self.task_id}
        log.info(f"query: {query}")
        document = {'$set': {'task_id': self.task_id, 'task_name': self.task_name, 'task_action': self.task_action}}
        result = client[self.save_dbname][self.save_collection].update_one(
            query,  # Query parameter
            document,
            upsert=True  # Options
        )
        log.info(f"Matched document with 'task_id' equals to '{self.task_id}': {result.matched_count}")
        log.info(f"Modified document with 'task_id' equals to '{self.task_id}': {result.modified_count}")

        final_time = time.time()
        log.info(f"Storing data in MongoDB total time: {((final_time - initial_time) / 60)} minutes")

    def _db_load(self) -> dict:
        """Load the task from the db"""
        log.info(f"Loading data (task_id: '{self.task_id}') from '{self.save_dbname}.{self.save_collection}' ...")
        initial_time = time.time()

        client: MongoClient = get_default_mongo_client()
        task = client[self.save_dbname][self.save_collection].find_one({'task_id': self.task_id}, {'_id': 0})

        final_time = time.time()
        log.info(f"Loading data from MongoDB total time: {((final_time - initial_time) / 60)} minutes")
        return task

    @staticmethod
    def save_dict_to_json(d, json_path):
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=4)


class AggregatedMetrics(object):
    def __init__(
            self,
            dbname: str = '',
            collection: str = '',
            task: str = '',
            corpus: str = ''
    ):
        self.save_dbname = dbname
        self.save_collection = collection
        self.task = task
        self.corpus = corpus

    def _get_experiments_query(self):
        # Get experiments from MongoDB.
        return {
            'task_id': {'$exists': True},
            'task_action': {'$exists': True},
            'task_action.kwargs': {'$exists': True},
            'task_action.train': {'$exists': True},
            'task_action.evaluation': {'$exists': True},
            'task_action.kwargs.scheduler': {'$exists': True},
            'task_action.kwargs.model': {'$exists': True},
            'task_action.kwargs.dataset': {'$exists': True},
            'task_action.train.model_meta_file': {'$exists': True},
            'task_action.evaluation.metrics': {'$exists': True},
            'task_action.kwargs.dataset.task': self.task,
            'task_action.kwargs.dataset.corpus': self.corpus,
        }

    @staticmethod
    def _get_experiments_projection():
        return {
            '_id': 0,
            'task_id': 1,
            'dataset': '$task_action.kwargs.dataset',
            'scheduler': '$task_action.kwargs.scheduler',
            'model': '$task_action.kwargs.model',
            'metrics': '$task_action.evaluation.metrics'
        }

    @staticmethod
    def _metric_name_exits(metric_name: list = None, metric_column_name: list = None):
        # Check if metric name exists.
        if metric_name is not None and len(metric_name) > 0:
            for name in metric_name:
                assert name in metric_column_name, 'Metric name not exists in saved experiments.'

    @staticmethod
    def _get_summary_fields(columns: list = None, metric_name: list = None, metric_column_name: list = None):
        result = {}
        for i, row in enumerate(columns):
            row_dict = row.copy()
            if metric_name is not None:                
                del row_dict['model_save_dir']
                if 'batch_size' in row_dict.keys():
                    del row_dict['batch_size']
                del row_dict['query_limit']
                del row_dict['dataset_save_dir']

                if metric_name is not None and len(metric_name) > 0:
                    for name in metric_column_name:
                        if name not in metric_name:
                            del row_dict[name]
            result[i] = row_dict
        return result

    def aggregate_metrics(
            self,
            metric_name: list = None,
            sort_by: list = None,
            num_models: int = 0,
            fields: list = None
    ) -> dict:
        """Aggregate the metrics of all experiments in MongoDB.
        Args:
            metric_name: (list) metrics names
            sort_by: (list) columns to sort by
            num_models: (int) limit of read results from MongoDB
            fields: (list) fields names
        Example:

            _aggregate_metrics(
                ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro'], ['accuracy'], 1
            )
        """
        query = self._get_experiments_query()
        projection = self._get_experiments_projection()

        # All results are retrieved to limit their number after sorting.
        df = load_dataframe_from_mongodb(
            database_name=self.save_dbname,
            collection_name=self.save_collection,
            query=query,
            projection=projection,
            query_limit=0
        )

        if df.empty:
            log.error(
                f"No metrics saved in '{self.save_dbname}.{self.save_collection}' for task '{self.task}' "
                f"and corpus '{self.corpus}'.")
            raise ValueError('Ensure database name, collection name, task name and corpus are correct.')

        # Explode MongoDB fields.
        df_dataset = pd.json_normalize(df['dataset'])
        df_scheduler = pd.json_normalize(df['scheduler'])
        df_model = pd.json_normalize(df['model'])
        df_metrics = pd.json_normalize(df['metrics'])
        exploded_df = pd.concat([df['task_id'], df_dataset, df_scheduler, df_model, df_metrics], axis=1)

        # Sort by
        if sort_by is not None and len(sort_by) > 0:
            for col in sort_by:
                assert col in exploded_df.columns, log.error(f"Column '{col}' not exists in saved experiments.")
            exploded_df = exploded_df.sort_values(by=sort_by, ascending=False)

        # Query limit
        if num_models > 0:
            exploded_df = exploded_df.head(num_models)

        # Check if metric name exists.
        df_metrics_columns = df_metrics.columns
        self._metric_name_exits(metric_name, list(df_metrics_columns))

        excluded_fields = []
        if fields is not None:
            for column in exploded_df.columns:
                if column not in fields + metric_name:
                    excluded_fields.append(column)

            exploded_df.drop(list(excluded_fields), axis=1, inplace=True)
            metric_name = None

        # Select fields and metrics
        exploded_df_columns = exploded_df.to_dict('records')
        result = self._get_summary_fields(exploded_df_columns, metric_name, list(df_metrics_columns))

        return result

    @staticmethod
    def _metrics_to_table(metrics):
        # Assumes everything has the same metrics
        headers = metrics[list(metrics.keys())[0]].keys()
        table = [[idx] + [values[h] for h in headers] for idx, values in metrics.items()]
        res = tabulate(table, headers, tablefmt='pretty')

        return res

    def results_summary(
            self,
            metric_name: list = None,
            sort_by: list = None,
            num_models: int = 0,
            fields: list = None
    ) -> None:
        metrics = self.aggregate_metrics(
            metric_name=metric_name,
            sort_by=sort_by,
            num_models=num_models,
            fields=fields
        )
        table_metrics = self._metrics_to_table(metrics)

        # Display the table to terminal
        print(table_metrics)


class GridSearch(AggregatedMetrics):
    def __init__(
            self,
            objective: list = None,
            parameter_space: dict = None
    ):
        super().__init__(
            dbname='tasks',
            collection='experiments',
            task=parameter_space['dataset']['task'],
            corpus=parameter_space['dataset']['corpus']
        )
        self.objective = objective
        self.parameter_space = parameter_space.copy()

    def _explode_param_space(self) -> list:
        self.parameter_space['alias'] = ''
        params = [self.parameter_space]
        hyperparam_groups = ['scheduler', 'model']
        for group in hyperparam_groups:
            for key in self.parameter_space[group]:
                if isinstance(self.parameter_space[group][key], list):
                    params_list = []
                    for val in self.parameter_space[group][key]:
                        for par in params:
                            tmp = copy.deepcopy(par)
                            tmp['alias'] = tmp['alias'] + '_' + key + '_' + str(val) if '' != tmp['alias'] \
                                else key + '_' + str(val)
                            tmp[group][key] = val
                            params_list.append(tmp)
                    params = params_list

        return params

    def search(self) -> None:
        """Launch training of the model with a set of hyperparameters"""
        # Perform hypersearch.
        search_params = self._explode_param_space()
        log.info(f"Dimension of parameter space: {len(search_params)}")

        for hyperparams in tqdm(search_params, total=len(search_params), desc='searches'):
            tic = time.time()

            # Modify reference_params param.
            hyperparams['reference_params'] = False

            # Create and save directory if not exists, and modify model_save_dir.
            params_path = str(Path(hyperparams['scheduler']['model_save_dir']) / hyperparams['alias'])
            if not os.path.exists(params_path):
                os.makedirs(params_path)
            hyperparams['scheduler']['model_save_dir'] = params_path

            model_builder = HyperModel(hyperparams=hyperparams)
            model_builder.run()

            log.info(f"Training and evaluating model total time = {((time.time() - tic) / 60)} minutes")

    def results_summary(
            self,
            metric_name: list = None,
            sort_by: list = None,
            num_models: int = 0,
            fields: list = None
    ) -> None:
        super().results_summary(
            metric_name=self.objective,
            sort_by=self.objective,
            num_models=num_models
        )

    @staticmethod
    def _get_best_models_projection():
        return {
            '_id': 0,
            'task_id': 1,
            'model_meta_file': '$task_action.train.model_meta_file',
            'metrics': '$task_action.evaluation.metrics'
        }

    def get_best_models(self, num_models=1):
        query = self._get_experiments_query()
        projection = self._get_best_models_projection()

        # All results are retrieved to limit their number after sorting.
        df = load_dataframe_from_mongodb(
            database_name=self.save_dbname,
            collection_name=self.save_collection,
            query=query,
            projection=projection,
            query_limit=0
        )

        if df.empty:
            log.error(
                f"No metrics saved in '{self.save_dbname}.{self.save_collection}' for task '{self.task}' "
                f"and corpus '{self.corpus}'.")
            raise ValueError('Ensure database name, collection name, task name and corpus are correct.')

        # Explode MongoDB fields.
        df_model_meta_file = df['model_meta_file']
        df_metrics = pd.json_normalize(df['metrics'])
        exploded_df = pd.concat([df['task_id'], df_model_meta_file, df_metrics], axis=1)

        # Check if metric name exists.
        df_metrics_columns = df_metrics.columns
        self._metric_name_exits(self.objective, list(df_metrics_columns))

        # Sort by
        exploded_df = exploded_df.sort_values(by=self.objective, ascending=False)

        # Query limit
        if num_models > 0:
            exploded_df = exploded_df.head(num_models)

        # Return id and model_meta_file.
        exploded_df.drop(list(df_metrics_columns), axis=1, inplace=True)
        return exploded_df.to_dict('records')


class Experiment(Task):
    def __init__(self, model_builder, test, w2i, word_embed, params):
        super().__init__()
        self.params = params.copy()
        self.kwargs = get_task_kwargs(self.params)
        self.task_id = self.get_hash()
        self.save_dbname = 'tasks'
        self.save_collection = 'experiments'
        self.task = self._db_load()

        # Check if task exists.
        assert self.task is not None, 'No task is loaded.'

        self.task_name = self.task['task_name'] if 'task_name' in self.task else None
        self.task_action = self.task['task_action'] if 'task_action' in self.task else None

        self.model_builder = model_builder
        self.test = test
        self.w2i = w2i
        self.word_embed = word_embed
        self.model_meta_file = self._get_model_meta_file()
        self.model = self.load()

    def _get_model_meta_file(self):
        model_meta_file = None
        if self.task_action is not None and 'train' in self.task_action \
                and 'model_meta_file' in self.task_action['train']:
            model_meta_file = self.task_action['train']['model_meta_file']

        if model_meta_file is None or model_meta_file == '':
            raise ValueError("Missing model meta file to load")
        return model_meta_file

    def load(self):
        if self.params['model']['architecture'] == 'tree_lstm':
            model = self.model_builder(
                n_classes=self.kwargs['model']['n_classes'],
                w2i=self.w2i,
                word_embed=self.word_embed,
                params=self.kwargs['model'],
                model_meta_file=self.model_meta_file
            )
        else:
            model = self.model_builder(
                params=self.kwargs,
                model_meta_file=self.model_meta_file
            )

        return model

    def run(self) -> None:
        raise NotImplementedError("This method must be inherited")


class HyperModel(Task):
    def __init__(self, hyperparams):
        super().__init__()
        self.params = hyperparams.copy()
        self.kwargs = {
            'scheduler': self.params['scheduler'],
            'model': self.params['model'],
            'dataset': self.params['dataset']
        }
        self.task_id = self.get_hash()
        self.task_name = f"{self.params['dataset']['task']}-{self.params['dataset']['corpus']}-" \
                         f"{self.params['model']['architecture']}-{self.params['alias']}"
        self.save_dbname = 'tasks'
        self.save_collection = 'experiments'
        self.task = self._db_load()
        self.task_action = {}

        if bool(self.task):
            if bool(self.task['task_action']):
                self.task_action = self.task['task_action']

    @staticmethod
    def _run_subprocess(cmd):
        try:
            completed = subprocess.run(cmd)
            return_code = completed.returncode
        except subprocess.CalledProcessError as err:
            print('ERROR:', err)
            return_code = -1
        else:
            print('Return code:', completed.returncode)

        return return_code

    def load(self):
        raise NotImplementedError("This method must be inherited")

    def run(self) -> None:
        """Launch training of the model with a set of hyperparameters"""
        # Write parameters in json file
        params_file = str(Path(self.params['scheduler']['model_save_dir']) / 'params.json')
        self.save_dict_to_json(self.params, params_file)

        # Defines Python executable.
        python_exe = os.environ.get('PYTHON_EXECUTABLE', sys.executable)

        # Launch training with this config
        execute_train = True
        if bool(self.task_action):
            if 'train' in self.task_action and not self.params['hyper_search_overwrite']:
                execute_train = False
                log.info(f"Matched training with 'task_id' equals to '{self.task_id}' and "
                         f"hyper_search_overwrite is {self.params['hyper_search_overwrite']}")

        if execute_train:
            module = f"syn.model.build.{self.params['model']['architecture']}.train"
            if self.params['dataset']['task'] == 'custom_assignation':
                module += '_custom_assignation'
            train_cmd = [python_exe, '-m', module, '--params_file', params_file]
            log.info(f"Running command: '{train_cmd}'.")
            result = self._run_subprocess(train_cmd)
            if result != 0:
                log.error(f"Error executing {train_cmd}")
                raise SystemExit()

        # Launch evaluation with this config
        execute_evaluation = True
        if bool(self.task_action):
            if 'evaluation' in self.task_action and not self.params['hyper_search_overwrite']:
                execute_evaluation = False
                log.info(f"Matched evaluation with 'task_id' equals to '{self.task_id}' and "
                         f"hyper_search_overwrite is {self.params['hyper_search_overwrite']}")

        if execute_evaluation:
            module = f"syn.model.build.{self.params['model']['architecture']}.evaluate"
            if self.params['dataset']['task'] == 'custom_assignation':
                module += '_custom_assignation'
            evaluate_cmd = [python_exe, '-m', module, '--params_file', params_file]
            log.info(f"Running command: '{evaluate_cmd}'.")
            result = self._run_subprocess(evaluate_cmd)
            if result != 0:
                log.error(f"Error executing {evaluate_cmd}")
                raise SystemExit()


class ConsensusFit(object):
    def __init__(
            self,
            database_name: str = 'tasks',
            collection_name: str = 'experiments',
            corpus: str = 'bugzilla',
            tasks_objectives: dict = None,
            common_hyperparams: List[List[str]] = None
    ):
        self.database_name = database_name
        self.collection_name = collection_name
        self.corpus = corpus
        self.tasks_objectives = tasks_objectives
        self.common_hyperparams = common_hyperparams
        self.experiments_results = self._get_experiments_results()
        self.specific_args = None

    def _get_experiments_results(self):
        tasks = list(self.tasks_objectives.keys())

        query = {
            'task_action': {'$exists': True},
            'task_action.kwargs': {'$exists': True},
            'task_action.evaluation': {'$exists': True},
            'task_action.kwargs.scheduler': {'$exists': True},
            'task_action.kwargs.model': {'$exists': True},
            'task_action.kwargs.dataset': {'$exists': True},
            'task_action.evaluation.metrics': {'$exists': True},
            'task_action.kwargs.dataset.corpus': self.corpus,
            'task_action.kwargs.dataset.task': {"$in": tasks}
        }
        log.info(f"query: {query}")
        projection = {
            '_id': 0,
            'kwargs': '$task_action.kwargs',
            'task': '$task_action.kwargs.dataset.task',
            'metrics': '$task_action.evaluation.metrics'
        }
        log.info(f"projection: {projection}")
        mongo_data = list(read_collection(
            database_name=self.database_name,
            collection_name=self.collection_name,
            query=query,
            projection=projection,
            query_limit=0
        ))

        results: List[List[Tuple[Any, Any]]] = []
        for task in tasks:
            data_aux = []
            for d in mongo_data:
                if d['task'] == task:
                    data_aux.append((d['kwargs'], d['metrics'][self.tasks_objectives[d['task']]]))

            if len(data_aux) > 0:
                results.append(data_aux)

        for i, task in enumerate(tasks):
            log.info(f"Experiments result dimension for task '{task}': {len(results[i])}")

        return results

    def get_common_hyperparameters_rank(self):
        return consensus.rank_common_parameters(self.experiments_results, self.common_hyperparams)

    def get_best_common_hyperparameters(self):
        return self.get_common_hyperparameters_rank()[0][0]

    def get_specific_hyperparameters_rank(self, hyperparameter):
        specific_parameters_rank = {}
        for result, task in zip(self.experiments_results, list(self.tasks_objectives.keys())):
            specific_parameters_rank[task] = consensus.rank_specific_parameters(result, hyperparameter)[0]

        return specific_parameters_rank

    def get_best_specific_hyperparameters_rank(self):
        return self.get_specific_hyperparameters_rank(self.get_best_common_hyperparameters())

    def common_hyperparameters_rank_summary(self, ) -> None:
        common_hyperparameters_rank = self.get_common_hyperparameters_rank()

        table_common_hyperparameters = []
        common_hyperparameters_header = []
        for list_element in common_hyperparameters_rank:
            common_hyperparameters_row = []
            for common_hyperparam in list_element[0]:
                common_hyperparameters_header.append(common_hyperparam[0])
                common_hyperparameters_row.append(common_hyperparam[1])
            table_common_hyperparameters.append(common_hyperparameters_row)

        print(tabulate(table_common_hyperparameters, headers=common_hyperparameters_header, tablefmt="pretty"))

    def task_losses_summary(self, ) -> None:
        common_hyperparameters_rank = self.get_common_hyperparameters_rank()

        table_task_losses_values = []
        for list_element in common_hyperparameters_rank:
            for i, task_loss in enumerate(list_element[1]):
                task_losses_row = [list(self.tasks_objectives)[i], task_loss]
                table_task_losses_values.append(task_losses_row)

        print(tabulate(table_task_losses_values, headers=['task', 'task_losses'], tablefmt="pretty"))

    def specific_hyperparameters_rank_summary(self) -> None:
        specific_hyperparameters_rank = self.get_best_specific_hyperparameters_rank()

        table_specific_hyperparameters = []
        specific_hyperparameters_header = ['task']
        metric_names = []
        for metric in list(self.tasks_objectives.values()):
            if metric not in metric_names:
                metric_names.append(metric)

        specific_hyperparameters_header.append('/'.join(metric_names))

        for dict_element in specific_hyperparameters_rank.items():
            specific_hyperparameters_row = [dict_element[0], dict_element[1][1]]
            for dict_element_value in dict_element[1][0].items():
                for param in dict_element_value[1].items():
                    specific_hyperparameters_header.append(f"{dict_element_value[0]}_{param[0]}")
                    specific_hyperparameters_row.append(param[1])

            table_specific_hyperparameters.append(specific_hyperparameters_row)

        print(tabulate(table_specific_hyperparameters, headers=specific_hyperparameters_header, tablefmt="pretty"))
