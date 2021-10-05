#!/usr/bin/env python3
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv

from definitions import ROOT_DIR, SYN_ENV
from syn.model.build.treelstm.vectorizer.AttentionVectorizer import get_attention_vector_raw_data
from syn.model.build.treelstm.vectorizer.ConstituencyParser import get_constituency_tree_raw_data
from syn.model.build.treelstm.vectorizer.VectorizedIssue import VectorizedIssue
from syn.model.build.treelstm.vectorizer.WordEmbedding import get_pretrained_embeddings


def get_vectorized_issue(db_name, col_name, embeddings_size=100, attention_vector=True, categorical=False, column=None):
    # Stores the start time of the method execution to calculate the time it takes.
    initial_time = time.time()
    # Defines logger.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    log.debug(f"\n[START OF EXECUTION]")

    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    pretrained_embeddings = get_pretrained_embeddings(embeddings_size)

    if not attention_vector:
        vectorized_issue = VectorizedIssue(
            pretrained_embeddings=pretrained_embeddings,
            constituency_trees_raw_data=get_constituency_tree_raw_data(db_name, col_name, categorical=categorical)
        )
    else:
        vectorized_issue = VectorizedIssue(
            pretrained_embeddings=pretrained_embeddings,
            attention_vector_raw_data=get_attention_vector_raw_data(db_name, col_name, categorical=categorical,
                                                                    column=column)
        )

    log.debug(f"\n[END OF EXECUTION]")
    final_time = time.time()
    log.info(f"Execution total time = {((final_time - initial_time) / 60)} minutes")

    return vectorized_issue
