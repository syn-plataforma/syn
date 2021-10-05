#!/usr/bin/env python3
import os
import time
from pathlib import Path
from typing import Any, Union, Iterable

import log4p
import numpy as np
from dotenv import load_dotenv
from numpy.core._multiarray_umath import ndarray

from definitions import ROOT_DIR, SYN_ENV
from syn.model.build.treelstm.vectorizer.PretrainedWordEmbedding import PretrainedWordEmbedding
from syn.model.build.treelstm.utils import cleanup


def convert_to_binary(embedding_path):
    f = open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []

    with open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            splitlines = line.split()
            vocab_write.write(splitlines[0].strip())
            vocab_write.write("\n")
            wv.append([float(val) for val in splitlines[1:]])
        count += 1

    np.save(embedding_path + ".npy", np.array(wv))


def load_embedding_matrix(embedding_path):
    print("Loading binary word embedding from {0}.vocab and {0}.npy".format(embedding_path))

    with open(embedding_path + '.vocab', 'r', encoding='utf-8') as f_in:
        index2word = [line.strip() for line in f_in]

    wv = np.load(embedding_path + '.npy')
    embedding_matrix = {}
    for i, w in enumerate(index2word):
        embedding_matrix[w] = wv[i]

    return embedding_matrix


def get_pretrained_embeddings(size=100, model='glove'):
    # Stores the start time of the method execution to calculate the time it takes.
    initial_time = time.time()
    # Defines logger.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    log.debug(f"\n[START OF EXECUTION]")

    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    # Vocabulary and pretrained word embeddings path.
    pretrained_embeddings = {
        'glove': {
            'file': Path(os.environ.get('WORD_EMBEDDINGS_PATH')) / model / f"glove.6B.{size}d.txt",
            'vocab': Path(os.environ.get('WORD_EMBEDDINGS_PATH')) / model / f"glove.vocab.6B.{size}d.txt",
            'embeddings': Path(os.environ.get('WORD_EMBEDDINGS_PATH')) / model / f"glove.embeddings.6B.{size}d.npy"
        }
    }

    pre_vocab = []
    with open(pretrained_embeddings[model]['vocab'], encoding='utf-8') as fin:
        log.info(f"Reading pretrained word embeddings vocabulary: '{pretrained_embeddings[model]['vocab']}'.")
        for line in fin:
            pre_vocab.append(cleanup(line.strip()))

    # Reverse vocabulary.
    pre_reverse_vocab: dict = {n: i for i, n in enumerate(pre_vocab)}

    # Word embeddings.
    log.info(f"Reading pretrained word embeddings vocabulary: '{pretrained_embeddings[model]['embeddings']}'.")
    pre_embs: Union[Union[ndarray, Iterable, int, float, tuple, dict], Any] = \
        np.load(pretrained_embeddings[model]['embeddings'])

    log.info("Loaded pretrained word embeddings.")

    log.debug(f"\n[END OF EXECUTION]")
    final_time = time.time()
    log.info(f"Execution total time = {((final_time - initial_time) / 60)} minutes")

    return PretrainedWordEmbedding(size=size, reversed_vocabulary=pre_reverse_vocab, embeddings=pre_embs)
