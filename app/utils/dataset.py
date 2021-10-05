import pandas as pd
from tqdm import tqdm

from syn.helpers.mongodb import load_dataframe_from_mongodb
from syn.helpers.nlp.trees import get_trees_from_string
from syn.helpers.task import get_trees_column_name, get_attention_vectors_column_name, \
    get_structured_data_column_name, get_task_dataset_projection, get_tokens_column_name


def get_pairs_dataset(
        arch: str = None,
        dataset: pd.DataFrame = None,
        task: str = '',
        corpus: str = '',
        query_limit: int = 0
) -> pd.DataFrame:
    projection = get_task_dataset_projection(arch=arch, task=task)
    # Query only non rejected documents
    query = {'rejected': False} if arch == 'tree_lstm' else {}
    df_task_dataset = load_dataframe_from_mongodb(
        database_name=corpus,
        collection_name=f"normalized_clear",
        query=query,
        projection=projection,
        sort_by='creation_ts',
        query_limit=query_limit
    )

    df_task_dataset['bug_id'] = pd.to_numeric(df_task_dataset['bug_id'])
    dataset_merged = dataset.merge(df_task_dataset, how='cross', suffixes=('_left', '_right'))

    return dataset_merged


def get_similarity_dataset(
        dataset: pd.DataFrame = None,
        corpus: str = '',
        near_issues: bool = False,
        query_limit: int = 0

) -> pd.DataFrame:
    collection_name = 'similar_pairs' if not near_issues else 'near_pairs'
    df_similar_pairs = load_dataframe_from_mongodb(
        database_name=corpus,
        collection_name=collection_name
    )

    # Sort by creation_ts
    df = df_similar_pairs.sort_values('creation_ts')

    if query_limit > 0:
        df = df.head(query_limit).copy()
    dataset_merged = df.merge(dataset, left_on='bug_id', right_on='bug_id',
                              suffixes=('_left', '_right'))
    return dataset_merged


def format_dataset(df: pd.DataFrame = None, task: str = '', corpus: str = '') -> list:
    trees_columns = get_trees_column_name(task, corpus)
    attention_vectors_columns = get_attention_vectors_column_name(task, corpus)
    structured_data_columns = get_structured_data_column_name(task, corpus)
    rows = []
    loop = tqdm(range(df.shape[0]), desc='rows')
    for i in loop:
        # if df.at[i, 'rejected']:
        #     continue
        row_elements = []
        # Columns with trees.
        for column_name in trees_columns:
            row_elements.append(get_trees_from_string(df.at[i, column_name]))

        # Columns with attention vectors.
        for column_name in attention_vectors_columns:
            row_elements.append(df.at[i, column_name])

        # Columns with structured data vectors.
        for column_name in structured_data_columns:
            row_elements.append(df.at[i, column_name])

        # bug_id for duplicity and similartiy tasks
        if task in ['duplicity', 'similarity']:
            row_elements.append(df.at[i, 'bug_id_right'])

        rows.append(tuple(row_elements))

    return rows if task in ['duplicity', 'similarity'] else rows[0]


def format_codebooks_dataset(df: pd.DataFrame = None, task: str = '', corpus: str = '') -> list:
    tokens_columns = get_tokens_column_name(task, corpus)
    structured_data_columns = get_structured_data_column_name(task, corpus)
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

        # bug_id for duplicity and similartiy tasks
        if task in ['duplicity', 'similarity']:
            row_elements.append(df.at[i, 'bug_id_right'])

        rows.append(tuple(row_elements))

    return rows if task in ['duplicity', 'similarity'] else rows[0]
