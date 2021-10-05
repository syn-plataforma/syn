"""Plot Accuracy vs Number of codewords"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import read_collection
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()

log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        description='Plot confusion matrix.'
    )
    parser.add_argument('--db_name', default='tasks', type=str, help='Database name.')
    parser.add_argument('--collection_name', default='experiments', type=str, help='Collection name.')
    parser.add_argument('--task', default='prioritization', type=str, help='Task name.')
    parser.add_argument('--corpus', default='openOffice', type=str, help='Corpus name.')
    parser.add_argument('--embeddings_model', default=None, type=str, help='Embeddings model.')
    parser.add_argument('--embeddings_pretrained', default=True, dest='embeddings_pretrained',
                        action='store_true', help="Pre-trained word embeddings.")
    parser.add_argument('--no_embeddings_pretrained', dest='embeddings_pretrained', action='store_false',
                        help="Untrained word embeddings.")
    parser.add_argument('--use_structured_data', default=True, dest='use_structured_data',
                        action='store_true', help="Use structured data.")
    parser.add_argument('--no_use_structured_data', dest='use_structured_data',
                        action='store_false', help="No use structured data.")
    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'task': args.task,
        'corpus': args.corpus,
        'embeddings_model': args.embeddings_model,
        'embeddings_pretrained': args.embeddings_pretrained,
        'use_structured_data': args.use_structured_data
    }


def get_data(field: str = None, field_list: list = None, query: dict = None) -> pd.DataFrame:
    # read accuracy and number of codewords from experiment
    log.info(f"Loading accuracy and number of codewords from experiments ...")
    tic = time.time()
    data = {}
    projection = {
        '_id': 0,
        'n_codewords': '$task_action.kwargs.model.codebooks_n_codewords',
        'accuracy': '$task_action.evaluation.metrics.accuracy'
    }
    for embeddings_size in field_list:
        query[f"task_action.kwargs.model.{field}"] = embeddings_size
        accuracy_n_codewords = read_collection(
            database_name=input_params['db_name'],
            collection_name=input_params['collection_name'],
            query=query,
            projection=projection,
            query_limit=0
        )

        assert accuracy_n_codewords, f"No data retrieved."

        n_codewords_data = [elem['n_codewords'] for elem in accuracy_n_codewords]
        accuracy_data = [elem['accuracy'] for elem in accuracy_n_codewords]
        if 'Number of codewords' not in data.keys():
            data['Number of codewords'] = n_codewords_data

        data[embeddings_size] = accuracy_data

    log.info(f"Loading accuracy and number of codewords from experiment "
             f"total time = {((time.time() - tic) / 60)} minutes")

    return pd.DataFrame.from_dict(data)


if __name__ == "__main__":
    log.info(f"Plotting accuracy vs number of codewords for embeddings size ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # embeddings sizes
    sizes = [100, 300]
    # classifiers
    # classifiers = ['decision_tree', 'random_forest', 'logistic_regression', 'extra_trees', 'c_support_vector']
    classifiers = ['decision_tree', 'random_forest', 'logistic_regression', 'extra_trees']

    for classifier in classifiers:
        embeddings_query = {
            "task_action.kwargs.dataset.corpus": input_params['corpus'],
            "task_action.kwargs.dataset.task": input_params['task'],
            "task_action.kwargs.model.classifier": classifier,
            "task_action.kwargs.model.embeddings_model": input_params['embeddings_model'],
            "task_action.kwargs.model.embeddings_pretrained": input_params['embeddings_pretrained']
        }

        # accuracy vs number of codewords for embeddings size
        embeddings_data = get_data(field='embeddings_size', field_list=sizes, query=embeddings_query)
        embeddings_df = pd.melt(embeddings_data, 'Number of codewords', var_name='Embeddings size',
                                value_name='Accuracy')
        embeddings_lp = sns.lineplot(
            x='Number of codewords', y='Accuracy', hue='Embeddings size', ci=None, marker='o', data=embeddings_df
        ).set(title=f"Classifier: {classifier}")
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()

        log.info(f"Plotting accuracy vs number of codewords for embeddings size "
                 f"total time = {((time.time() - initial_time) / 60)} minutes")

    # accuracy vs number of codewords for embeddings size
    for size in sizes:
        log.info(f"Plotting accuracy vs number of codewords for classifier model ...")
        classifier_query = {
            "task_action.kwargs.dataset.corpus": input_params['corpus'],
            "task_action.kwargs.dataset.task": input_params['task'],
            "task_action.kwargs.model.embeddings_model": input_params['embeddings_model'],
            "task_action.kwargs.model.embeddings_size": size,
            "task_action.kwargs.model.embeddings_pretrained": input_params['embeddings_pretrained']
        }
        classifier_data = get_data(field='classifier', field_list=classifiers, query=classifier_query)
        classifier_df = pd.melt(classifier_data, 'Number of codewords', var_name='Classifier model',
                                value_name='Accuracy')
        classifier_lp = sns.lineplot(
            x='Number of codewords', y='Accuracy', hue='Classifier model', ci=None, marker='o', data=classifier_df
        ).set(title=f"Embeddings size: {size}")
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()

        log.info(f"Plotting accuracy vs number of codewords for classifier model "
                 f"total time = {((time.time() - initial_time) / 60)} minutes")

    log.info(f"MODULE EXECUTED.")
