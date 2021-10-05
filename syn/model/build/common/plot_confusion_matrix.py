"""Plot confusion matrix"""

import argparse
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import read_collection
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.task import get_task_label_codes

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
    parser.add_argument('--task_id', default=None, type=str, help='Task identifier.')
    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'task': args.task,
        'corpus': args.corpus,
        'task_id': args.task_id
    }


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == "__main__":
    log.info(f"Plotting confusion matrix ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."
    assert input_params['task_id'] is not None and input_params['task_id'] != '', f"No task_id provided."

    # read confusion matrix from experiment
    log.info(f"Loading confusion matrix from experiment ...")
    tic = time.time()
    query = {"task_id": input_params['task_id'], 'task_action.evaluation.metrics.confusion_matrix': {'$exists': True}}
    projection = {'_id': 0, 'confusion_matrix': '$task_action.evaluation.metrics.confusion_matrix'}
    confusion_matrix_field = read_collection(
        database_name=input_params['db_name'],
        collection_name=input_params['collection_name'],
        query=query,
        projection=projection,
        query_limit=0
    )
    log.info(f"Loading confusion matrix from experiment total time = {((time.time() - tic) / 60)} minutes")

    # read label codes for task and corpus
    label_codes = get_task_label_codes(task=input_params['task'], corpus=input_params['corpus'], near=False)

    assert confusion_matrix_field and confusion_matrix_field[0] and confusion_matrix_field[0][
        'confusion_matrix'], f"No confusion matrix provided."

    plot_confusion_matrix(cm=np.array(confusion_matrix_field[0]['confusion_matrix']),
                          normalize=False,
                          target_names=label_codes.keys(),
                          title="Confusion Matrix")

    log.info(f"Plotting confusion matrix total time = {((time.time() - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
