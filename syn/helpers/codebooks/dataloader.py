import os
import pickle
import random
import time
from pathlib import Path
from typing import Union, MutableSequence, List, Tuple, Any

import pandas as pd
from tqdm import tqdm

from syn.helpers.logging import set_logger
from syn.helpers.mongodb import load_dataframe_from_mongodb

# Logger.
log = set_logger()


def get_task_model_projection(task: str = 'duplicity', model: str = 'tree_lstm', ) -> dict:
    task_model_projection = {
        'prioritization': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'tokens': 1,
                'structured_data': 1
            },
            'codebooks': {
                '_id': 0,
                'label': 1,
                'tokens': 1,
                'structured_data': 1
            }
        },
        'classification': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'tokens': 1,
                'structured_data': 1
            },
            'codebooks': {
                '_id': 0,
                'label': 1,
                'tokens': 1,
                'structured_data': 1
            }
        },
        'assignation': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'tokens': 1,
                'structured_data': 1
            },
            'codebooks': {
                '_id': 0,
                'label': 1,
                'tokens': 1,
                'structured_data': 1
            }
        },
        'duplicity': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'tokens_left': 1,
                'tokens_right': 1,
                'structured_data_left': 1,
                'structured_data_right': 1
            },
            'codebooks': {
                '_id': 0,
                'label': 1,
                'tokens_left': 1,
                'tokens_right': 1,
                'structured_data_left': 1,
                'structured_data_right': 1
            }
        },
        'similarity': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'tokens_left': 1,
                'tokens_right': 1,
                'structured_data_left': 1,
                'structured_data_right': 1
            },
            'codebooks': {
                '_id': 0,
                'label': 1,
                'tokens_left': 1,
                'tokens_right': 1,
                'structured_data_left': 1,
                'structured_data_right': 1
            }
        }
    }

    return task_model_projection[task][model]


def read_dataset_from_mongodb(
        database_name: str = 'bugzilla',
        collection_name: str = 'duplicity_task_train_dataset',
        query: dict = None,
        projection: dict = None,
        tokens_columns: list = None,
        structured_data_columns: list = None,
        query_limit: int = 0
) -> List[Tuple[Union[list, Any], ...]]:
    # List[Tuple[list, Any, Any]]
    # Read MongoDB collection.
    df = load_dataframe_from_mongodb(database_name=database_name, collection_name=collection_name, query=query,
                                     projection=projection, query_limit=query_limit)

    log.info(f"Generating codebooks ...")
    tokens_columns = tokens_columns if tokens_columns is not None and len(tokens_columns) > 0 else ['tokens']
    structured_data_columns = structured_data_columns \
        if structured_data_columns is not None and len(structured_data_columns) > 0 else ['structured_data']
    rows = []
    loop = tqdm(range(df.shape[0]), desc='rows')
    for i in loop:
        row_elements = []
        # Columns with tokens.
        for column_name in tokens_columns:
            row_elements.append(df.at[i, column_name])

        # Columns with structured data vectors.
        for column_name in structured_data_columns:
            row_elements.append(df.at[i, column_name])

        # Column with label.
        row_elements.append(df.at[i, 'label'])

        rows.append(tuple(row_elements))

    return rows


def read_dataset(
        model: str = 'tree_lstm',
        task: str = 'duplicity',
        corpus: str = 'bugzilla',
        dataset_name: str = 'train',
        tokens_columns: list = None,
        structured_data_columns: list = None,
        query_limit: int = 0,
        save_dir: str = ''
) -> Union[None, pd.DataFrame, MutableSequence]:
    # Check if exists a saved version of dataset.
    path = Path(save_dir) / f"{model}_{dataset_name}_data.pkl"
    # TODO: Revisar por qué aparecen los OR en la siguiente línea.
    if not os.path.exists(path) or corpus == 'eclipse' or corpus == 'bugzilla':
        data = read_dataset_from_mongodb(
            database_name=corpus,
            collection_name=f"{task}_task_{dataset_name}_dataset",
            projection=get_task_model_projection(task, model),
            tokens_columns=tokens_columns,
            structured_data_columns=structured_data_columns,
            query_limit=query_limit
        )
    else:
        tic = time.time()
        log.info(f"Reading data from filesystem: '{path}'.")
        with open(path, 'rb') as f:
            saved_data = pickle.load(f)
        log.info(f"Reading data from filesystem total time: {(time.time() - tic) / 60} minutes")
        return saved_data

    # TODO: Revisar por qué aparecen los OR en la siguiente línea.
    if corpus != 'eclipse' and corpus != 'bugzilla':
        log.info(f"Saving data to filesystem: '{path}'.")
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data


class DataLoader(object):
    def __init__(
            self,
            model: str = 'tree_lstm',
            task: str = 'duplicity',
            corpus: str = 'bugzilla',
            dataset_name: str = 'duplicity_task_train_dataset',
            tokens_columns: list = None,
            structured_data_columns: list = None,
            query_limit: int = 0,
            save_dir: str = ''
    ):
        self.data = read_dataset(model=model, task=task, corpus=corpus, dataset_name=dataset_name,
                                 tokens_columns=tokens_columns, structured_data_columns=structured_data_columns,
                                 query_limit=query_limit, save_dir=save_dir)
        self.n_samples = len(self.data)
        self.idx = 0
        self.reset()

    def reset(self, shuffle=True):
        self.idx = 0
        if shuffle:
            random.shuffle(self.data)

    def __iter__(self):
        while self.idx < self.n_samples:
            yield self.data[self.idx]
            self.idx += 1

    def batches(self, batch_size=25):
        while self.idx < self.n_samples:
            yield self.data[self.idx: self.idx + batch_size]
            self.idx += batch_size
