"""Plot constituency trees"""

import argparse
import os
import time

from nltk.tree import Tree

from syn.helpers.argparser import dataset_parser
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import read_collection
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()

log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(
        parents=[dataset_parser],
        description='Plot constituency trees inserted in MongoDB.'
    )

    args = parser.parse_args()

    return {
        'corpus': args.corpus,
        'dataset_name': args.dataset_name,
        'query_limit': args.query_limit
    }


if __name__ == "__main__":
    log.info(f"Plotting constituency trees ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    # Loads dataset.
    log.info(f"Loading constituency trees ...")
    query = {}
    projection = {'_id': 0, 'trees': '$collapsed_binary_constituency_trees'}
    df_trees = read_collection(
        database_name=input_params['corpus'],
        collection_name=input_params['dataset_name'],
        query=query,
        projection=projection,
        query_limit=input_params['query_limit']
    )
    log.info(f"Constituency trees loaded.")

    log.info(f"Plotting constituency trees ...")
    for description_trees in df_trees:
        for tree in description_trees['trees']:
            tree_obj = Tree.fromstring(tree)
            tree_obj.draw()

    log.info(f"Plotting constituency trees total time = {((time.time() - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
