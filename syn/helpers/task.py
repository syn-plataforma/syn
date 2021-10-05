import os

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import read_collection

load_environment_variables()
log = set_logger()


def get_task_dataset_projection(arch: str = 'tree_lstm', corpus: str = 'bugzilla', task: str = None) -> dict:
    fields = {
        'prioritization': {
            '_id': 0,
            'bug_id': 1,
            'product': 1,
            'bug_severity': 1,
            'label': '$priority',
            'component': 1,
            'tokens': '$detailed_tokens'
        },
        'classification': {
            '_id': 0,
            'bug_id': 1,
            'product': 1,
            'priority': 1,
            'label': '$bug_severity',
            'component': 1,
            'tokens': '$detailed_tokens'
        },
        'duplicity': {
            '_id': 0,
            'bug_id': 1,
            'product': 1,
            'bug_severity': 1,
            'priority': 1,
            'component': 1,
            'tokens': '$detailed_tokens'
        },
        'assignation': {
            '_id': 0,
            'bug_id': 1,
            'product': 1,
            'bug_severity': 1,
            'priority': 1,
            'component': 1,
            'tokens': '$detailed_tokens'
        },
        'similarity': {
            '_id': 0,
            'bug_id': 1,
            'product': 1,
            'bug_severity': 1,
            'priority': 1,
            'component': 1,
            'tokens': '$detailed_tokens'
        }
    }

    if corpus != 'bugzilla' and (task == 'assignation' or task == 'similarity'):
        raise NotImplementedError(
            f"Corpus '{corpus}' only works with prioritization, classification and duplicity tasks."
        )

    if arch == 'tree_lstm':
        fields[task]['trees'] = '$collapsed_binary_constituency_trees'
        fields[task]['attention_vectors.0'] = '$constituents_embeddings'

    if corpus == 'bugzilla' and task == 'assignation':
        fields[task]['label'] = '$assigned_to'

    return fields[task]


def get_task_dataset_query(arch: str = 'tree_lstm', task: str = None) -> dict:
    query = {
        'prioritization': {
            'priority': {
                '$exists': 'true'
            },
            'detailed_tokens.0.0': {
                '$exists': 'true'
            }
        },
        'classification': {
            'bug_severity': {
                '$exists': 'true'
            },
            'detailed_tokens.0.0': {
                '$exists': 'true'
            }
        },
        'duplicity': {
            'detailed_tokens.0.0': {
                '$exists': 'true'
            }
        },
        'assignation': {
            'detailed_tokens.0.0': {
                '$exists': 'true'
            },
            'assigned_to': {
                '$exists': 'true'
            }
        },
        'similarity': {
            'detailed_tokens.0.0': {
                '$exists': 'true'
            }
        }
    }

    if arch == 'tree_lstm':
        query[task]['collapsed_binary_constituency_trees.0'] = {'$exists': 'true'}
        query[task]['constituents_embeddings.0'] = {'$exists': 'true'}

    return query[task]


def get_task_structured_data_codes(corpus: str, column: str) -> dict:
    column = column.replace('_left', '').replace('_right', '')
    collection_name = f"{column}_codes"

    mongo_data = read_collection(corpus, collection_name)
    result = {}
    for doc in mongo_data:
        result[doc[column]] = doc[f"{column}_code"]
    return result


def get_task_label_codes(task: str, corpus: str, near: bool) -> dict:
    label_column_name = get_label_column_name(task, corpus)
    if '' == label_column_name:
        raise NotImplementedError('This module only works with prioritization, classification and assignation tasks.')

    collection_name = f"{label_column_name}_codes"

    if 'duplicity' == task:
        collection_name = f"pairs_{label_column_name}_codes"

    if 'similarity' == task:
        if not near:
            collection_name = f"similar_pairs_{label_column_name}_codes"
        else:
            collection_name = f"near_pairs_{label_column_name}_codes"

    log.info(f"Label column name for task '{task}' and corpus '{corpus}': '{label_column_name}'")

    mongo_data = read_collection(corpus, collection_name)
    result = {}
    for doc in mongo_data:
        result[doc[label_column_name]] = doc[f"{label_column_name}_code"]
    return result


def get_number_classes(task: str, corpus: str, near: bool) -> int:
    label_codes = get_task_label_codes(task, corpus, near)
    return len(label_codes.keys())


def get_number_structured_data_columns(task: str) -> int:
    number_structured_data_columns = {
        'prioritization': len(os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')) - 1,
        'classification': len(os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')) - 1,
        'assignation': len(os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')),
        'duplicity': len(os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')),
        'similarity': len(os.environ['STRUCTURED_DATA_COLUMN_NAMES'].split(',')),
    }

    return number_structured_data_columns[task]


def get_label_collection_name(task: str = 'duplicity', corpus: str = 'bugzilla') -> str:
    collection_name = {
        'prioritization': {
            'openOffice': 'normalized_clear',
            'netBeans': 'normalized_clear',
            'eclipse': 'normalized_clear',
            'bugzilla': 'normalized_clear',
        },
        'classification': {
            'openOffice': 'normalized_clear',
            'netBeans': 'normalized_clear',
            'eclipse': 'normalized_clear',
            'bugzilla': 'normalized_clear',
        },
        'assignation': {
            'openOffice': 'normalized_clear',
            'netBeans': 'normalized_clear',
            'eclipse': 'normalized_clear',
            'bugzilla': 'normalized_clear'
        },
        'duplicity': {
            'openOffice': 'pairs',
            'netBeans': 'pairs',
            'eclipse': 'pairs',
            'bugzilla': 'pairs'
        },
        'similarity': {
            'openOffice': 'pairs',
            'netBeans': 'pairs',
            'eclipse': 'pairs',
            'bugzilla': 'pairs'
        }
    }
    label_collection_name = collection_name[task][corpus]
    return label_collection_name


def get_label_column_name(task: str = 'duplicity', corpus: str = 'bugzilla') -> str:
    column_name = {
        'prioritization': {
            'openOffice': 'priority',
            'netBeans': 'priority',
            'eclipse': 'priority',
            'bugzilla': 'priority',
        },
        'classification': {
            'openOffice': 'bug_severity',
            'netBeans': 'bug_severity',
            'eclipse': 'bug_severity',
            'bugzilla': 'bug_severity',
        },
        'assignation': {
            'openOffice': '',
            'netBeans': '',
            'eclipse': '',
            'bugzilla': 'assigned_to'
        },
        'custom_assignation': {
            'openOffice': '',
            'netBeans': '',
            'eclipse': '',
            'bugzilla': 'assigned_to'
        },
        'duplicity': {
            'openOffice': 'dec',
            'netBeans': 'dec',
            'eclipse': 'dec',
            'bugzilla': 'dec'
        },
        'similarity': {
            'openOffice': '',
            'netBeans': '',
            'eclipse': '',
            'bugzilla': 'dec'
        }
    }
    label_column_name = column_name[task][corpus]
    if '' == label_column_name:
        raise NotImplementedError("Assignation and similarity is only implemented for bugzilla corpus.")
    return label_column_name


def get_tokens_column_name(task: str = 'duplicity', corpus: str = 'bugzilla') -> list:
    column_name = {
        'prioritization': {
            'openOffice': ['tokens'],
            'netBeans': ['tokens'],
            'eclipse': ['tokens'],
            'bugzilla': ['tokens'],
        },
        'classification': {
            'openOffice': ['tokens'],
            'netBeans': ['tokens'],
            'eclipse': ['tokens'],
            'bugzilla': ['tokens'],
        },
        'assignation': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['tokens']
        },
        'custom_assignation': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['tokens']
        },
        'duplicity': {
            'openOffice': ['tokens_left', 'tokens_right'],
            'netBeans': ['tokens_left', 'tokens_right'],
            'eclipse': ['tokens_left', 'tokens_right'],
            'bugzilla': ['tokens_left', 'tokens_right']
        },
        'similarity': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['tokens_left', 'tokens_right']
        }
    }
    tokens_column_name = column_name[task][corpus]
    if len(tokens_column_name) == 0:
        raise NotImplementedError("Assignation and similarity is only implemented for bugzilla corpus.")
    return tokens_column_name


def get_trees_column_name(task: str = 'duplicity', corpus: str = 'bugzilla') -> list:
    column_name = {
        'prioritization': {
            'openOffice': ['trees'],
            'netBeans': ['trees'],
            'eclipse': ['trees'],
            'bugzilla': ['trees'],
        },
        'classification': {
            'openOffice': ['trees'],
            'netBeans': ['trees'],
            'eclipse': ['trees'],
            'bugzilla': ['trees'],
        },
        'assignation': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['trees']
        },
        'duplicity': {
            'openOffice': ['trees_left', 'trees_right'],
            'netBeans': ['trees_left', 'trees_right'],
            'eclipse': ['trees_left', 'trees_right'],
            'bugzilla': ['trees_left', 'trees_right']
        },
        'similarity': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['trees_left', 'trees_right']
        }
    }
    trees_column_name = column_name[task][corpus]
    if len(trees_column_name) == 0:
        raise NotImplementedError("Assignation and similarity is only implemented for bugzilla corpus.")
    return trees_column_name


def get_attention_vectors_column_name(task: str = 'duplicity', corpus: str = 'bugzilla') -> list:
    column_name = {
        'prioritization': {
            'openOffice': ['attention_vectors'],
            'netBeans': ['attention_vectors'],
            'eclipse': ['attention_vectors'],
            'bugzilla': ['attention_vectors'],
        },
        'classification': {
            'openOffice': ['attention_vectors'],
            'netBeans': ['attention_vectors'],
            'eclipse': ['attention_vectors'],
            'bugzilla': ['attention_vectors'],
        },
        'assignation': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['attention_vectors'],
        },
        'duplicity': {
            'openOffice': ['attention_vectors_left', 'attention_vectors_right'],
            'netBeans': ['attention_vectors_left', 'attention_vectors_right'],
            'eclipse': ['attention_vectors_left', 'attention_vectors_right'],
            'bugzilla': ['attention_vectors_left', 'attention_vectors_right'],
        },
        'similarity': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['attention_vectors_left', 'attention_vectors_right'],
        }
    }
    attention_vectors_name = column_name[task][corpus]
    if len(attention_vectors_name) == 0:
        raise NotImplementedError("Assignation and similarity is only implemented for bugzilla corpus.")
    return attention_vectors_name


def get_structured_data_column_name(task: str = 'duplicity', corpus: str = 'bugzilla') -> list:
    column_name = {
        'prioritization': {
            'openOffice': ['structured_data'],
            'netBeans': ['structured_data'],
            'eclipse': ['structured_data'],
            'bugzilla': ['structured_data'],
        },
        'classification': {
            'openOffice': ['structured_data'],
            'netBeans': ['structured_data'],
            'eclipse': ['structured_data'],
            'bugzilla': ['structured_data'],
        },
        'assignation': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['structured_data'],
        },
        'custom_assignation': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['structured_data'],
        },
        'duplicity': {
            'openOffice': ['structured_data_left', 'structured_data_right'],
            'netBeans': ['structured_data_left', 'structured_data_right'],
            'eclipse': ['structured_data_left', 'structured_data_right'],
            'bugzilla': ['structured_data_left', 'structured_data_right'],
        },
        'similarity': {
            'openOffice': [],
            'netBeans': [],
            'eclipse': [],
            'bugzilla': ['structured_data_left', 'structured_data_right'],
        }
    }
    structured_data_columns = column_name[task][corpus]
    if len(structured_data_columns) == 0:
        raise NotImplementedError("Assignation and similarity is only implemented for bugzilla corpus.")
    return structured_data_columns


def get_task_features_column_names(task: str = None, architecture: str = 'tree_lstm') -> list:
    fields = {
        'tree_lstm': {
            'prioritization': ['product', 'bug_severity', 'component', 'trees', 'attention_vectors'],
            'classification': ['product', 'priority', 'component', 'trees', 'attention_vectors'],
            'duplicity': ['bug_id', 'product', 'bug_severity', 'priority', 'component', 'trees', 'attention_vectors'],
            'assignation': ['product', 'bug_severity', 'priority', 'component', 'trees', 'attention_vectors'],
            'similarity': ['bug_id', 'product', 'bug_severity', 'priority', 'component', 'trees', 'attention_vectors']
        },
        'codebooks': {
            'prioritization': ['product', 'bug_severity', 'component', 'tokens'],
            'classification': ['product', 'priority', 'component', 'tokens'],
            'duplicity': ['bug_id', 'product', 'bug_severity', 'priority', 'component', 'tokens'],
            'assignation': ['product', 'bug_severity', 'priority', 'component', 'tokens'],
            'custom_assignation': ['product', 'bug_severity', 'priority', 'component', 'tokens', 'composite_data',
                                   'label'],
            'similarity': ['bug_id', 'product', 'bug_severity', 'priority', 'component', 'tokens']
        }
    }
    return fields[architecture][task]


def get_task_request_params_names(task: str) -> list:
    fields = {
        'prioritization': ['product', 'bug_severity', 'component'],
        'classification': ['product', 'priority', 'component'],
        'duplicity': ['bug_id', 'product', 'bug_severity', 'priority', 'component'],
        'assignation': ['product', 'bug_severity', 'priority', 'component'],
        'custom_assignation': ['product', 'bug_severity', 'priority', 'component'],
        'similarity': ['bug_id', 'product', 'bug_severity', 'priority', 'component']
    }
    return fields[task]


def get_aggregation_project_exclusion_fields():
    return {
        '_id': 0,
        'dup_id': 0,
        'version': 0,
        'delta_ts': 0,
        'bug_status': 0,
        'resolution': 0,
        'normalized_short_desc': 0,
        'normalized_description': 0,
        'binary_constituency_trees': 0,
        'constituency_trees': 0,
        'detailed_constituent_embeddings': 0,
        'detailed_lemmas': 0,
        'detailed_tokens': 0,
        'embeddings_context_branch': 0,
        'num_sentences': 0,
        'sentences': 0,
        'token_branch': 0,
        'tokens': 0,
        'total_num_tokens': 0
    }
