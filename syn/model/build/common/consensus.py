"""Consensus module"""
import os
from collections import abc

import numpy as np
from pymongo import MongoClient

from syn.helpers.mongodb import get_default_local_mongo_client, get_default_mongo_client


def _flatten_dict(d, parent_key='', sep='_'):
    """Flatten a dict with nested dicts. flatten_dict({"a":{"b":1, "c":1}, "c": 2}) -> {'a_b': 1, 'a_c': 1, 'c': 2}"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_common_kwargs_configurations(task_values, common_kwargs):
    """Get a flattened list of common_kwargs configurations available in test_values """
    common_kwargs_flatten = ["_".join(x) for x in common_kwargs]
    common_kwargs_values = {tuple([(k, v) for k, v in _flatten_dict(value[0]).items() if k in common_kwargs_flatten])
                            for value in task_values}
    common_kwargs_values = [dict(x) for x in common_kwargs_values]
    return common_kwargs_values


# Get values for given common_args
def _get_values(task_values, common_kwargs_flatten):
    """Get the subset of task_values which meet the flattened kwarg restriction given"""
    d1 = task_values
    return [value for value in d1 if
            {k: v for k, v in _flatten_dict(value[0]).items() if k in common_kwargs_flatten} == common_kwargs_flatten]


def _get_task_losses(task_values, common_kwargs_flatten):
    """Get the amount of loss in task_values due to picking the configuration in common_kwargs_flatten"""
    best = max(x[1] for x in task_values)
    restricted_best = max(x[1] for x in _get_values(task_values, common_kwargs_flatten))
    return best - restricted_best


def _get_task_preferences(task_values, common_kwargs):
    """Get a dict mapping each available common_kwargs configuration available to its loss"""
    return dict((tuple(kwarg.items()), _get_task_losses(task_values, kwarg)) for kwarg in
                _get_common_kwargs_configurations(task_values, common_kwargs))


def rank_common_parameters(tasks_values, common_kwargs, weights=None):
    """
    Get an overall common parameter ranking

    Args:
        tasks_values (list of list of (dict, float)): For each of the tasks, a list with 2-tuples relating kwargs
                                                     (as a dict) and values.
        common_kwargs (list of list of str): Arguments which are adjusted for all the tasks.
        weights (list of float): Weights for each of the tasks.

    Returns:
        list of 2-tuples: Ordered list of preferred common configurations in the sense of the consensus algorithm, each
                          of them as a 2-tuple with the flattened common arguments and a list with the values for each
                          of the tasks.

    """
    preferences = [_get_task_preferences(x, common_kwargs) for x in tasks_values]
    # List of dict to dict of list
    v = {k: [dic[k] for dic in preferences] for k in preferences[0]}
    # Return sorted by ascending sum of losses
    return sorted(list(v.items()), key=lambda x: np.average(x[1], weights=weights))


def rank_specific_parameters(task_value, common_kwargs_flatten):
    """
    Get a task-specific parameter ranking
    Args:
        task_value (list of (dict, float)): A list with 2-tuples relating kwargs and values.
        common_kwargs_flatten (list of (str, float)): A list of the common kwargs and their values.

    Returns:
        list of 2-tuples: Ordered list of preferred configurations for the task with the fixed common parameters, each
                          of them as a configuration dict and the value of the task.
    """
    return sorted(list(_get_values(task_value, dict(common_kwargs_flatten))), key=lambda x: x[1], reverse=True)


def task_hyper_fit_0(tasks, corpus, date_range_train, date_range_test, common_kwargs):
    """
    Fit the hyperparameters of a set of run tasks (from the task module)

    Args:
        tasks (dict): A mapping from task names to its relevant metric
        corpus (str): The corpus used to select the tasks to take into account.
        date_range_train (2-tuple of 3-tuple of int): Training date range to take into account.
        date_range_test (2-tuple of 3-tuple of int): Test date range to take into account.
        common_kwargs (list of list of str): List of hyperparameters to adjust in common.

    Returns:
        - Tuple of 2-tuple: Pairs of parameters (with their names flattned) and their values.
        - dict of str to (dict, float): A mapping from task names to their configuration and the score.

    """

    client: MongoClient = get_default_mongo_client() if os.environ['WORK_ENVIRONMENT'] == 'aws' \
        else get_default_local_mongo_client()

    task_names = list(tasks.keys())

    data = list(client["incidences"]["tasks"].find({'type': {"$in": task_names},
                                                    'kwargs.corpus': corpus,
                                                    'kwargs.date_range_train': date_range_train,
                                                    'kwargs.date_range_test': date_range_test,

                                                    }))

    results = [[(d["kwargs"], d["results"][tasks[d["type"]]]) for d in data if d["type"] == task_type] for task_type in
               task_names]

    common_rank = rank_common_parameters(results, common_kwargs)

    best = common_rank[0][0]

    return best, {task: rank_specific_parameters(result, best)[0] for result, task in zip(results, tasks)}


def task_hyper_fit(tasks, corpus, common_hyperparams):
    """
    Fit the hyperparameters of a set of run tasks (from the task module)

    Args:
        tasks (dict): A mapping from task names to its relevant metric
        corpus (str): The corpus used to select the tasks to take into account.
        common_hyperparams (list of list of str): List of hyperparameters to adjust in common.

    Returns:
        - Tuple of 2-tuple: Pairs of parameters (with their names flattned) and their values.
        - dict of str to (dict, float): A mapping from task names to their configuration and the score.

    """

    client: MongoClient = get_default_mongo_client() if os.environ['WORK_ENVIRONMENT'] == 'aws' \
        else get_default_local_mongo_client()

    task_names = list(tasks.keys())

    data = list(client["incidences"]["tasks"].find({'type': {"$in": task_names},
                                                    'kwargs.corpus': corpus
                                                    }))

    results = [[(d["kwargs"], d["results"][tasks[d["type"]]]) for d in data if d["type"] == task_type] for task_type in
               task_names]

    common_rank = rank_common_parameters(results, common_hyperparams)

    best = common_rank[0][0]

    return best, {task: rank_specific_parameters(result, best)[0] for result, task in zip(results, tasks)}
