"""Read, split and save the dataset for SYN model"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.dataset import get_input_params, load_dataset, save_dataset, encode_dataset_labels, \
    balance_data, build_duplicity_dataset, build_assignation_dataset, build_similarity_dataset, \
    encode_dataset_structured_data

load_environment_variables()
log = set_logger()

if __name__ == "__main__":
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Load normalized_clear dataset.
    log.info(f"Loading Dataframe ...")
    dataset = load_dataset(
        arch=input_params.arch,
        task=input_params.task,
        corpus=input_params.corpus,
        name=input_params.dataset_name,
        query_limit=input_params.query_limit
    )
    log.info(f"Dataset columns: {dataset.columns}")
    log.info(f"Dataframe loaded.")

    # if task is 'duplicity' need do more actions.
    if 'duplicity' == input_params.task:
        dataset = build_duplicity_dataset(dataset, input_params.corpus)

    # if task is 'assignation' need do more actions.
    if 'assignation' == input_params.task:
        dataset = build_assignation_dataset(dataset, input_params.corpus, int(input_params.n_developers))

    # if task is 'similarity' need do more actions.
    if 'similarity' == input_params.task:
        dataset = build_similarity_dataset(
            dataset,
            input_params.corpus,
            'similar_pairs' if not input_params.near_issues else 'near_pairs'
        )

    # Encode structured info.
    log.info(f"Encoding structured data ...")
    dataset = encode_dataset_structured_data(dataset, input_params.corpus, input_params.task)

    # Encode label.
    log.info(f"Encoding label ...")
    dataset = encode_dataset_labels(dataset, input_params.task, input_params.corpus, input_params.near_issues)

    label_value_counts = dataset['label'].value_counts()
    log.info(f"Label values counts after checking minimum number o members: {dict(dataset['label'].value_counts())}")

    # Balance data.
    if input_params.balance_data:
        dataset = balance_data(dataset, label_value_counts)

    # Split the dataset into train, dev and split. Make sure to always shuffle with a fixed seed so that the split is
    # reproducible.
    log.info(f"Splitting dataset in train, dev and test ...")
    y = dataset['label']
    X = dataset.drop(['label'], axis=1)
    test_size = int(float(os.environ['TEST_DATA_SIZE']) * len(dataset))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size if input_params.arch == 'tree_lstm' else test_size * 2,
        random_state=230,
        stratify=y
    )

    # task name
    task_name = input_params.task
    if 'similarity' == input_params.task:
        if input_params.near_issues:
            task_name = 'near_similarity'

    # Tree-LSTM needs dev dataset too
    if input_params.arch == 'tree_lstm':
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train,
            y_train,
            test_size=int(float(os.environ['DEV_DATA_SIZE']) * len(dataset)),
            random_state=230,
            stratify=y_train
        )
        dev_dataset = pd.concat([X_dev, y_dev], axis=1)
        log.info(f"Number of rows in dev dataset: {len(dev_dataset)}")

        inserted_dev_documents = save_dataset(
            task=task_name,
            corpus=input_params.corpus,
            dest='balanced_dev' if input_params.balance_data else 'dev',
            dataset=dev_dataset
        )
        assert inserted_dev_documents == len(dev_dataset)

    # Concat datasets to save in MongoDB.
    train_dataset = pd.concat([X_train, y_train], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)

    log.info(f"Number of rows in train dataset: {len(train_dataset)}")
    log.info(f"Number of rows in test dataset: {len(test_dataset)}")

    # Save the datasets to MongoDB.
    inserted_train_documents = save_dataset(
        task=task_name,
        corpus=input_params.corpus,
        dest='balanced_train' if input_params.balance_data else 'train',
        dataset=train_dataset
    )
    assert inserted_train_documents == len(train_dataset)

    inserted_test_documents = save_dataset(
        task=task_name,
        corpus=input_params.corpus,
        dest='balanced_test' if input_params.balance_data else 'test',
        dataset=test_dataset
    )
    assert inserted_test_documents == len(test_dataset)
    log.info(f"MODULE EXECUTED.")
