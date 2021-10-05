import argparse
import os
import time
from typing import Union

import pandas as pd
from pymongo import MongoClient, errors

from syn.helpers.argparser import dataset_parser, assignation_task_parser, similarity_task_parser, sentence_model_parser
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client, save_dataframe_to_mongodb, load_dataframe_from_mongodb
from syn.helpers.task import get_task_dataset_projection, get_task_dataset_query, get_task_label_codes, \
    get_task_structured_data_codes

# Logger.
log = set_logger()


class DatasetParams:

    def __init__(
            self,
            arch='tree_lstm',
            task='duplicates',
            corpus='bugzilla',
            dataset_name='normalized_clear',
            balance_data=True,
            query_limit=0,
            n_developers=30,
            near_issues=False
    ):
        self.arch = arch
        self.task = task
        self.corpus = corpus
        self.dataset_name = dataset_name
        self.balance_data = balance_data
        self.query_limit = query_limit
        self.n_developers = n_developers
        self.near_issues = near_issues


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[dataset_parser, assignation_task_parser, similarity_task_parser, sentence_model_parser],
        description='Read, split and save the dataset for SYN model.'
    )

    args = parser.parse_args()

    return DatasetParams(
        arch=args.architecture,
        task=args.task,
        corpus=args.corpus,
        dataset_name=args.dataset_name,
        balance_data=args.balance_data,
        query_limit=args.query_limit,
        n_developers=args.n_developers,
        near_issues=args.near_issues,
    )


def get_datasets() -> list:
    return ['train', 'dev', 'evaluation']


def load_dataset_from_mongodb(
        arch: str = 'tree_lstm',
        task: str = 'duplicity',
        database_name: str = 'bugzilla',
        collection_name: str = 'normalized_clear',
        query_limit: int = 0
) -> pd.DataFrame:
    log.info(f"Reading data from MongoDB: '{database_name}.{collection_name}'")

    tic = time.time()
    # MongoClient connection.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[database_name]
    col = db[collection_name]

    if col.name not in db.list_collection_names():
        raise errors.CollectionInvalid(f"Collection '{db.name}.{col.name}' not found. "
                                       f"Make sure your collection name is correct.")

    # Queries MongoDB collection.
    query = get_task_dataset_query(arch, task)
    log.info(f"Query filter document: {query}")
    projection = get_task_dataset_projection(arch, database_name, task)
    log.info(f"Projection document: {projection}")

    pipeline = [
        {'$match': query},
        {'$project': projection}
    ]

    if query_limit > 0:
        pipeline.append({'$limit': query_limit})

    mongodb_data = col.aggregate(pipeline)

    # Expands cursor and builds DataFrame.
    df = pd.DataFrame(list(mongodb_data))
    log.info(f"Read documents from MongoDB: {len(df.index)}")
    log.info(f"Reading data from MongoDB total time: {(time.time() - tic) / 60} minutes")

    return df


def load_dataset(
        arch: str = 'tree_lstm',
        task: str = 'duplicity',
        corpus: str = 'bugzilla',
        name: str = 'eclipse_all_duplicate_det_task',
        query_limit: int = 0
) -> Union[None, pd.DataFrame]:
    return load_dataset_from_mongodb(
        arch=arch, task=task, database_name=corpus, collection_name=name, query_limit=query_limit
    )


def save_dataset(
        task: str = 'duplicity',
        corpus: str = 'bugzilla',
        dest: str = 'train',
        dataset: pd.DataFrame = None
) -> Union[None, int]:
    return save_dataframe_to_mongodb(
        database_name=corpus,
        collection_name=f"{task}_task_{dest}_dataset",
        dataframe=dataset
    )


def check_stratify(dataset: pd.DataFrame = None, label_value_counts: pd.Series = None) -> None:
    # Check minimum number of groups for label to stratify. We need 2 members for first stratification an another 2
    # members for second stratification.
    checked_dataset = dataset.copy()
    for index, value in label_value_counts.items():
        if value < 4:
            log.error(f"The least populated label has only {value} members, which is too few to stratify. "
                      f"The minimum number of groups for any class cannot be less than 4.")
            log.info(f"Removing rows with label equals to {index} ...")
            log.info(f"Ensure to update number of classes in  model train, validate and test.")
            checked_dataset = checked_dataset[checked_dataset['label'] != index]

    return checked_dataset


def encode_dataset_labels(
        dataset: pd.DataFrame = None,
        task: str = 'duplicity',
        corpus: str = 'bugzilla',
        near: bool = False
) -> pd.DataFrame:
    labels = get_task_label_codes(task, corpus, near)

    dataset['label'] = dataset['label'].apply(lambda x: labels[x])
    label_value_counts = dataset['label'].value_counts()
    log.info(f"Label values counts: {dict(label_value_counts)}")
    # Check minimum number of groups for label to stratify.
    check_stratify(dataset, label_value_counts)

    return dataset


def structured_data_to_array(x, columns, suffix):
    structured_data_array = []
    for column in columns:
        if '' != suffix:
            if column.find(suffix) != -1:
                structured_data_array.append([float(x[column])])
        else:
            structured_data_array.append([float(x[column])])
    return structured_data_array


def encode_dataset_structured_data(
        dataset: pd.DataFrame = None,
        corpus: str = 'bugzilla',
        task: str = 'duplicity'
) -> pd.DataFrame:
    structured_data_column_name = os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')
    pairs_structured_data_column_name = []
    for column_name in structured_data_column_name:
        pairs_structured_data_column_name.append(f"{column_name}_left")
        pairs_structured_data_column_name.append(f"{column_name}_right")

    updated_structured_data_column_name = structured_data_column_name + pairs_structured_data_column_name

    if len(updated_structured_data_column_name) == 0:
        raise ValueError('No structured data column names defined.')
    log.info(f"Structured data column name: {updated_structured_data_column_name}")

    structured_columns = []
    for column in updated_structured_data_column_name:
        if column in dataset.columns:
            log.info(f"Encoding columna '{column}' ...")
            codes = get_task_structured_data_codes(corpus, column)
            dataset[column] = dataset[column].apply(lambda x: codes[x])
            label_value_counts = dataset['label'].value_counts()
            log.info(f"Label values counts: {dict(label_value_counts)}")
            structured_columns.append(column)

            # Check minimum number of groups for label to stratify.
            check_stratify(dataset, label_value_counts)

    if task in ['duplicity', 'similarity']:
        dataset['structured_data_left'] = dataset.apply(
            lambda x: structured_data_to_array(x, structured_columns, '_left'),
            axis=1
        )
        dataset['structured_data_right'] = dataset.apply(
            lambda x: structured_data_to_array(x, structured_columns, '_right'),
            axis=1
        )
    else:
        dataset['structured_data'] = dataset.apply(
            lambda x: structured_data_to_array(x, structured_columns, ''),
            axis=1
        )

    return dataset


def balance_data(dataset: pd.DataFrame = None, label_value_counts: pd.Series = None) -> pd.DataFrame:
    log.info(f"Balancing data (under-sampling) ...")
    # Class count
    min_size = label_value_counts.min()

    # Divide by class
    dataset_by_class = []
    for label in label_value_counts.keys().sort_values():
        dataset_by_class.append(dataset[dataset['label'] == label].sample(min_size))

    # under - sampling
    dataset = pd.concat(dataset_by_class, axis=0).copy()
    log.info(f"Label values counts: {dict(dataset['label'].value_counts())}")

    return dataset


def build_duplicity_dataset(dataset: pd.DataFrame = None, corpus: str = 'bugzilla') -> pd.DataFrame:
    # Load pairs dataframe.
    df_pairs = load_dataframe_from_mongodb(
        database_name=corpus,
        collection_name='pairs'
    )

    # Check duplicated pairs.
    log.info(f"Looking for duplicates in pair 'bug1' - 'bug2' ...")
    df_pairs['bug1-bug2'] = df_pairs.apply(lambda x: f"{x['bug1']}-{x['bug2']}", axis=1)
    df_pairs['bug2-bug1'] = df_pairs.apply(lambda x: f"{x['bug2']}-{x['bug1']}", axis=1)
    log.info(f"Rows before drop duplicates: {df_pairs.shape[0]}")
    df_pairs.drop_duplicates(subset='bug1-bug2', keep=False, inplace=True)
    log.info(f"Rows after drop duplicates: {df_pairs.shape[0]}")

    log.info(f"Looking for duplicates 'bug1' - 'bug2' equals to 'bug2' - 'bug1' ...")
    df_pairs['duplicated'] = df_pairs.apply(lambda x: x['bug1-bug2'] == x['bug2-bug1'], axis=1)
    log.info(f"Rows with duplicates pairs: {df_pairs[df_pairs['duplicated']].shape[0]}")

    df_pairs_final = df_pairs[df_pairs['duplicated'] == False].copy()
    log.info(f"Rows after drop all types of duplicates: {df_pairs_final.shape[0]}")

    # Change bug_id column type.
    dataset['bug_id'] = pd.to_numeric(dataset['bug_id'])

    # Join on column bug1 and bug_id.
    df_pairs_bug1_dataset_bug_id = df_pairs_final.merge(dataset, left_on='bug1', right_on='bug_id')

    # Join on column bug2 and bug_id.
    result = df_pairs_bug1_dataset_bug_id.merge(dataset, left_on='bug2', right_on='bug_id',
                                                suffixes=('_left', '_right'))

    result.drop(['bug1', 'bug2', 'bug1-bug2', 'bug2-bug1', 'duplicated'], axis=1, inplace=True)

    # Rename column dec as label.
    result.rename(columns={"dec": "label"}, errors="raise", inplace=True)

    return result


def encode_and_save_assigned_to(dataset, corpus, n_developers):
    df = pd.DataFrame(columns=['assigned_to'])
    column_value_counts = dataset['label'].value_counts()

    df['assigned_to'] = column_value_counts.keys().to_list()

    # Assigning numerical values and storing in another column
    df[f"assigned_to_code"] = df['assigned_to'].index

    log.info(f"Assigned to codes: ")
    log.info(df[:n_developers])

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[corpus]

    col = db[f"assigned_to_codes"]

    if col.name in db.list_collection_names():
        db.drop_collection(col.name)

    log.info(f"Inserting documents ...")

    inserted_documents = col.insert_many(df.to_dict("records"))
    log.info(f"Inserted documents: {len(inserted_documents.inserted_ids)}")


def build_assignation_dataset(
        dataset: pd.DataFrame = None,
        corpus: str = 'bugzilla',
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

    encode_and_save_assigned_to(dataset_top_n_developers, corpus, n_developers)

    return dataset_top_n_developers


def build_similarity_dataset(
        dataset: pd.DataFrame = None,
        corpus: str = 'bugzilla',
        collection_name: str = 'similar_pairs'
) -> pd.DataFrame:
    # Load df_similar_pairs dataframe.
    df_similar_pairs = load_dataframe_from_mongodb(
        database_name=corpus,
        collection_name=collection_name
    )

    # Change bug_id column type.
    dataset['bug_id'] = pd.to_numeric(dataset['bug_id'])

    # Join on column bug1 and bug_id.
    df_pairs_bug1_dataset_bug_id = df_similar_pairs.merge(dataset, left_on='bug1', right_on='bug_id')

    # Join on column bug2 and bug_id.
    result = df_pairs_bug1_dataset_bug_id.merge(dataset, left_on='bug2', right_on='bug_id',
                                                suffixes=('_left', '_right'))

    result.drop(['bug1', 'bug2'], axis=1, inplace=True)

    # Rename column dec as label.
    result.rename(columns={"dec": "label"}, errors="raise", inplace=True)

    return result
