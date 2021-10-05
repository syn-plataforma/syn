import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from pymongo import MongoClient

from syn.helpers.assignation import get_normalized_cost, get_hash, get_composite_fields, get_resolution_time
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client
from syn.helpers.mongodb import save_dataframe_to_mongodb
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Search useful issues.')

    parser.add_argument('--db_name', default='bugzilla', type=str, help='Bugzilla database name.')
    parser.add_argument('--collection_name', default='clear', type=str, help='Bugzilla collection name.')
    parser.add_argument('--year', default='2019', type=str, help='Year to consider incidence as closed.')
    parser.add_argument('--month', default='11', type=str, help='Month to consider incidence as closed.')
    parser.add_argument('--day', default='01', type=str, help='Day to consider incidence as closed.')
    parser.add_argument('--time_delta_days_opened', default=30, type=int, help='Days to consider incidence as opened.')
    parser.add_argument('--n_developers', default=30, type=int, help='Number of developers')
    parser.add_argument('--composite_id_fields', default='component,product,priority', help='Composite ID fields.')
    parser.add_argument('--batch_size', default=10000, type=int, help='Batch size.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'year': args.year,
        'month': args.month,
        'day': args.day,
        'time_delta_days_opened': args.time_delta_days_opened,
        'n_developers': args.n_developers,
        'composite_id_fields': args.composite_id_fields,
        'batch_size': args.batch_size
    }


def build_n_developers_assignation_dataset(
        dataset: pd.DataFrame = None,
        n_developers: int = 30
) -> pd.DataFrame:
    # Group_by "assigned_to" and sort desc.
    assigned_to_value_counts = dataset['label'].value_counts().sort_values(ascending=False)
    df_top_n_developers = pd.DataFrame(
        data=assigned_to_value_counts[:n_developers].keys().to_list(),
        columns=['label']
    )
    log.info(f"Top k developers and incidences assigned to: ")
    log.info(assigned_to_value_counts[:n_developers])

    # Join on column 'assigned_to'.
    dataset_top_n_developers = df_top_n_developers.merge(dataset, left_on='label', right_on='label')

    return dataset_top_n_developers


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    log.info(f"Building train and test datasets ...")

    # MongoDB client.
    mongodb_client: MongoClient = get_default_mongo_client()

    # load Bugzilla issues
    db = mongodb_client[os.environ.get('BUGZILLA_MONGODB_DATABASE_NAME', input_params['db_name'])]
    collection = db[input_params['collection_name']]

    projection = {
        '_id': 0,
        'bug_id': 1,
        'product': 1,
        'priority': 1,
        'component': 1,
        'bug_severity': 1,
        'assigned_to': 1,
        'creation_ts': 1,
        'delta_ts': 1
    }

    # bugzilla dataset
    log.info(f"Reading data from '{db.name}.{collection.name}' ...")
    tic = time.time()
    query = {}
    train_data = collection.find(query, projection)
    df_base = pd.DataFrame(list(train_data))

    log.info(f"Reading data from '{db.name}.{collection.name}' "
             f"total execution time = {((time.time() - tic) / 60)} minutes")

    # Check empty Dataframe.
    if 0 == df_base.shape[0]:
        raise ValueError(f"No documents have been retrieved from "
                         f"'{db.name}.{collection.name}' collection.")

    df_base.rename(columns={'assigned_to': 'label'}, errors='raise', inplace=True)

    # get most assigned n_developers
    if input_params['n_developers'] > 0:
        log.info(f"Filtering data by '{input_params['n_developers']}' most assigned developers ...")
        df_base_n_developers = build_n_developers_assignation_dataset(df_base, input_params['n_developers']).copy()
        log.info(f"Total issues assigned to first {input_params['n_developers']} developers:"
                 f" {df_base_n_developers.shape[0]}")
        df_base = df_base_n_developers
        log.info(f"Filtering data by {input_params['n_developers']} most assigned developers "
                 f"total execution time = {((time.time() - tic) / 60)} minutes")

    # train dataset
    log.info(f"Filtering data by date ...")
    tic = time.time()
    # filter by date
    datetime_str = f"{input_params['year']}-{input_params['month']}-{input_params['day']}T00:00:00Z"
    datetime_closed_filter = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")
    df_train = df_base[df_base['delta_ts'] < datetime_closed_filter].copy()
    log.info(f"Total issues in train dataset: {df_train.shape[0]}")
    log.info(f"Filtering data by date total execution time = {((time.time() - tic) / 60)} minutes")

    # Check empty Dataframe.
    if 0 == df_train.shape[0]:
        raise ValueError(f"No documents have been recovered from '{db.name}.{collection.name}' collection, with a "
                         f"closing date of less than '{datetime_closed_filter}'.")

    # composite id fields
    composite_id_fields = input_params['composite_id_fields'].split(",") if '' != input_params[
        'composite_id_fields'] else os.environ["COMPOSITE_ID_FIELDS"].split(",")
    log.info(f"Calculating composite id ...")
    tic = time.time()
    df_train['composite_id'] = df_train[composite_id_fields].apply(lambda x: get_hash(*x), axis=1)
    log.info(f"Calculating composite id total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Calculating composite data ...")
    tic = time.time()
    df_train['composite_data'] = df_train[composite_id_fields].apply(lambda x: get_composite_fields(*x), axis=1)
    log.info(f"Calculating composite data total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Calculating resolution time in hours ...")
    tic = time.time()
    df_train['resolution_time_hours'] = df_train[['delta_ts', 'creation_ts']].apply(lambda x: get_resolution_time(*x),
                                                                                    axis=1)
    log.info(f"Calculating resolution time in hours total execution time = {((time.time() - tic) / 60)} minutes")

    # normalized cost dataset
    log.info(f"Calculating issues normalized cost ...")
    tic = time.time()
    df_normalized_cost = get_normalized_cost(df_train)
    normalized_cost_dict = {}
    for index, row in df_normalized_cost.iterrows():
        normalized_cost_dict[row['composite_id']] = row['normalized_cost']
    log.info(f"Calculating issue normalized cost total execution time = {((time.time() - tic) / 60)} minutes")

    # add normalized cost to train dataset
    log.info(f"Adding normalized cost ...")
    tic = time.time()
    df_train['normalized_cost'] = df_train['composite_id'].apply(lambda x: normalized_cost_dict[x])
    log.info(f"Adding normalized cost total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Inserting train dataset documents ...")
    train_name_suffix = 'lt_' + datetime_closed_filter.strftime("%Y_%m_%d")
    if input_params['n_developers'] > 0:
        train_name_suffix = f"{input_params['n_developers']}_" + train_name_suffix
    inserted_train_docs_number = save_dataframe_to_mongodb(
        database_name=input_params['db_name'],
        collection_name=f"assignation_task_custom_train_dataset_{train_name_suffix}",
        dataframe=df_train
    )
    log.info(f"Train dataset documents inserted: {inserted_train_docs_number}")

    # opened dataset
    log.info(f"Filtering data by date ...")
    tic = time.time()
    # filter by date
    datetime_opened_filter = datetime_closed_filter + timedelta(days=input_params['time_delta_days_opened'])
    df_opened = df_base[(df_base['delta_ts'] < datetime_opened_filter) & (
            df_base['delta_ts'] >= datetime_closed_filter)].copy()
    log.info(f"Total issues in opened dataset: {df_opened.shape[0]}")
    log.info(f"Filtering data by date total execution time = {((time.time() - tic) / 60)} minutes")

    # Check empty Dataframe.
    if 0 == df_opened.shape[0]:
        raise ValueError(f"No documents have been recovered from '{db.name}.{collection.name}' collection, with a "
                         f"closing date of less than '{datetime_opened_filter}' and "
                         f"greater than or equal '{datetime_closed_filter}'.")

    log.info(f"Calculating composite id ...")
    tic = time.time()
    df_opened['composite_id'] = df_opened[['component', 'product', 'priority']].apply(lambda x: get_hash(*x), axis=1)
    log.info(f"Calculating composite id total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Calculating resolution time ...")
    tic = time.time()
    df_opened['resolution_time_hours'] = df_opened[['delta_ts', 'creation_ts']].apply(lambda x: get_resolution_time(*x),
                                                                                      axis=1)
    log.info(f"Calculating resolution time total execution time = {((time.time() - tic) / 60)} minutes")

    # add normalized cost to opened dataset
    log.info(f"Adding normalized cost ...")
    tic = time.time()
    df_opened['normalized_cost'] = df_opened['composite_id'].apply(
        lambda x: normalized_cost_dict[x] if x in normalized_cost_dict.keys() else 0.0
    )
    log.info(f"Adding normalized cost total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Inserting opened dataset documents ...")
    opened_name_suffix = datetime_opened_filter.strftime("%Y_%m_%d")
    inserted_opened_docs_number = save_dataframe_to_mongodb(
        database_name=input_params['db_name'],
        collection_name=f"assignation_task_custom_opened_dataset_{train_name_suffix.replace('lt_', 'gte_')}_"
                        f"lt_{opened_name_suffix}",
        dataframe=df_opened
    )
    log.info(f"Opened dataset documents inserted: {inserted_opened_docs_number}")

    # test dataset
    log.info(f"Filtering data by date ...")
    tic = time.time()
    # filter by date
    df_test = df_base[df_base['delta_ts'] >= datetime_opened_filter].copy()
    log.info(f"Total issues in opened dataset: {df_test.shape[0]}")
    log.info(f"Filtering data by date total execution time = {((time.time() - tic) / 60)} minutes")

    # Check empty Dataframe.
    if 0 == df_test.shape[0]:
        raise ValueError(f"No documents have been recovered from '{db.name}.{collection.name}' collection, with a "
                         f"closing date of greater than or equal '{datetime_opened_filter}'.")

    log.info(f"Calculating composite id ...")
    tic = time.time()
    df_test['composite_id'] = df_test[['component', 'product', 'priority']].apply(lambda x: get_hash(*x), axis=1)
    log.info(f"Calculating composite id total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Calculating composite data ...")
    tic = time.time()
    df_test['composite_data'] = df_test[composite_id_fields].apply(lambda x: get_composite_fields(*x), axis=1)
    log.info(f"Calculating composite data total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Calculating resolution time ...")
    tic = time.time()
    df_test['resolution_time_hours'] = df_test[['delta_ts', 'creation_ts']].apply(lambda x: get_resolution_time(*x),
                                                                                  axis=1)
    log.info(f"Calculating resolution time total execution time = {((time.time() - tic) / 60)} minutes")

    # add normalized cost to test dataset
    log.info(f"Adding normalized cost ...")
    tic = time.time()
    df_test['normalized_cost'] = df_test['composite_id'].apply(
        lambda x: normalized_cost_dict[x] if x in normalized_cost_dict.keys() else 0.0
    )
    log.info(f"Adding normalized cost total execution time = {((time.time() - tic) / 60)} minutes")

    log.info(f"Inserting test dataset documents ...")
    test_name_suffix = 'gte_' + opened_name_suffix
    if input_params['n_developers'] > 0:
        test_name_suffix = f"{input_params['n_developers']}_" + test_name_suffix
    inserted_test_docs_number = save_dataframe_to_mongodb(
        database_name=input_params['db_name'],
        collection_name=f"assignation_task_custom_test_dataset_{test_name_suffix}",
        dataframe=df_test
    )
    log.info(f"Test dataset documents inserted: {inserted_test_docs_number}")

    final_time = time.time()
    log.info(f"Building train and test datasets total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
