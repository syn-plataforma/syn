import os
import pickle
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Union, Iterator, MutableSequence, List, Tuple, Any

import pandas as pd
from tqdm import tqdm

from syn.helpers.logging import set_logger
from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.treelstm.attention import get_part_of_speech_weight

# Logger.
log = set_logger()


def get_trees_projection() -> dict:
    return {
        '_id': 0,
        'label': 1,
        'trees': 1
    }


def get_tokens_projection() -> dict:
    return {
        '_id': 0,
        'label': 1,
        'tokens': 1
    }


def get_task_model_projection(task: str = 'duplicity', model: str = 'tree_lstm', ) -> dict:
    task_model_projection = {
        'prioritization': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'trees': 1,
                'attention_vectors': 1,
                'structured_data': 1
            }
        },
        'classification': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'trees': 1,
                'attention_vectors': 1,
                'structured_data': 1
            }
        },
        'assignation': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'trees': 1,
                'attention_vectors': 1,
                'structured_data': 1
            }
        },
        'duplicity': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'trees_left': 1,
                'attention_vectors_left': 1,
                'trees_right': 1,
                'attention_vectors_right': 1,
                'structured_data_left': 1,
                'structured_data_right': 1
            }
        },
        'similarity': {
            'tree_lstm': {
                '_id': 0,
                'label': 1,
                'trees_left': 1,
                'attention_vectors_left': 1,
                'trees_right': 1,
                'attention_vectors_right': 1,
                'structured_data_left': 1,
                'structured_data_right': 1
            }
        }
    }

    return task_model_projection[task][model]


def get_trees_from_mongodb(row: list) -> list:
    return [Tree.from_string(tree) for tree in row]


def read_dataset_from_mongodb(
        database_name: str = 'bugzilla',
        collection_name: str = 'duplicity_task_train_dataset',
        query: dict = None,
        projection: dict = None,
        trees_columns: list = None,
        attention_vectors_columns: list = None,
        structured_data_columns: list = None,
        query_limit: int = 0
) -> List[Tuple[Union[list, Any], ...]]:
    # List[Tuple[list, Any, Any]]
    # Read MongoDB collection.
    df = load_dataframe_from_mongodb(database_name=database_name, collection_name=collection_name, query=query,
                                     projection=projection, query_limit=query_limit)

    log.info(f"Generating trees ...")
    trees_columns = trees_columns if trees_columns is not None and len(trees_columns) > 0 else ['trees']
    attention_vectors_columns = attention_vectors_columns \
        if attention_vectors_columns is not None and len(attention_vectors_columns) > 0 else ['attention_vectors']
    structured_data_columns = structured_data_columns \
        if structured_data_columns is not None and len(structured_data_columns) > 0 else ['structured_data']
    rows = []
    loop = tqdm(range(df.shape[0]), desc='rows')
    for i in loop:
        row_elements = []
        # Columns with trees.
        for column_name in trees_columns:
            row_elements.append(get_trees_from_mongodb(df.at[i, column_name]))

        # Columns with attention vectors.
        for column_name in attention_vectors_columns:
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
        trees_columns: list = None,
        attention_vectors_columns: list = None,
        structured_data_columns: list = None,
        query_limit: int = 0,
        save_dir: str = ''
) -> Union[None, pd.DataFrame, MutableSequence]:
    # Check if exists a saved version of dataset.
    path = Path(save_dir) / f"{model}_{dataset_name}_data.pkl"
    if not os.path.exists(path) or corpus == 'eclipse' or corpus == 'bugzilla':
        data = read_dataset_from_mongodb(
            database_name=corpus,
            collection_name=f"{task}_task_{dataset_name}_dataset",
            projection=get_task_model_projection(task, model),
            trees_columns=trees_columns,
            attention_vectors_columns=attention_vectors_columns,
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

    if corpus != 'eclipse' and corpus != 'bugzilla':
        log.info(f"Saving data to filesystem: '{path}'.")
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data


def get_vocabs(trees):
    label_vocab = Counter()
    word_vocab = Counter()
    for tree in trees:
        label_vocab.update([n.label for n in tree.non_terms()])
        word_vocab.update([leaf.label for leaf in tree.leaves()])
    words = ["_UNK_"] + [x for x, c in word_vocab.items() if c > 0]
    w2i = {w: i for i, w in enumerate(words)}
    return w2i, words


def _tokenize_string(s):
    tokker = re.compile(r" +|[()]|[^ ()]+")
    toks = [t for t in [match.group(0) for match in tokker.finditer(s)] if t[0] != " "]
    return toks


def _within_bracket(toks: Iterator[str]):
    label = next(toks)
    children = []
    for tok in toks:
        if tok == "(":
            children.append(_within_bracket(toks))
        elif tok == ")":
            return Tree(label, children)
        else:
            children.append(Tree(tok, None))
    raise RuntimeError('Error Parsing tree string')


class Tree(object):
    def __init__(self, label, children=None):
        self.label = label if children is None else get_part_of_speech_weight(label)
        self.children = children

    @staticmethod
    def from_string(string: str):
        toks = iter(_tokenize_string(string))
        if next(toks) != "(":
            raise RuntimeError('Error Parsing tree string.')
        return _within_bracket(toks)

    def __str__(self):
        if self.children is None:
            return self.label
        return "[%s %s]" % (self.label, " ".join([str(c) for c in self.children]))

    def is_leaf(self):
        return self.children is None

    def leaves_iter(self):
        if self.is_leaf():
            yield self
        else:
            for c in self.children:
                for leaf in c.leaves_iter():
                    yield leaf

    def leaves(self):
        return list(self.leaves_iter())

    def non_terms_iter(self):
        if not self.is_leaf():
            yield self
            for c in self.children:
                for n in c.non_terms_iter():
                    yield n

    def non_terms(self):
        return list(self.non_terms_iter())


class DataLoader(object):
    def __init__(
            self,
            model: str = 'tree_lstm',
            task: str = 'duplicity',
            corpus: str = 'bugzilla',
            dataset_name: str = 'duplicity_task_train_dataset',
            trees_columns: list = None,
            attention_vectors_columns: list = None,
            structured_data_columns: list = None,
            query_limit: int = 0,
            save_dir: str = ''
    ):
        self.data = read_dataset(model=model, task=task, corpus=corpus, dataset_name=dataset_name,
                                 trees_columns=trees_columns, attention_vectors_columns=attention_vectors_columns,
                                 structured_data_columns=structured_data_columns, query_limit=query_limit,
                                 save_dir=save_dir)
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
