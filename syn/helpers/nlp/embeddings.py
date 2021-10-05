"""Read, filter and save word embeddings for SYN model"""

import argparse
import codecs
import time
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger

load_environment_variables()
log = set_logger()


class WordEmbeddingsParams:
    def __init__(
            self,
            corpus='bugzilla',
            task='duplicates',
            model="glove",
            size=300
    ):
        self.corpus = corpus
        self.task = task
        self.model = model
        self.size = size


def get_input_params():
    parser = argparse.ArgumentParser(
        description='Read, filter and save word embeddings for SYN model.'
    )
    # task
    parser.add_argument(
        '--task',
        default='duplicity',
        type=str,
        choices=['assignation', 'classification', 'duplicity', 'prioritization', 'similarity'],
        help='Task for which the word embeddings are read, filtered or saved.'
    )

    # corpus and database name
    parser.add_argument(
        '--corpus',
        default='bugzilla',
        type=str,
        choices=['bugzilla', 'eclipse', 'netBeans', 'openOffice'],
        help='Corpus for which the word embeddings are read, filtered or saved.'
    )
    # model
    parser.add_argument(
        '--model',
        default='glove',
        type=str,
        choices=['glove', 'word2vec', 'fasttext', 'fine-tuned'],
        help='Word embeddings model.'
    )

    # size
    parser.add_argument(
        '--size',
        default=300,
        type=int,
        choices=[100, 300],
        help='Word embeddings size.'
    )

    args = parser.parse_args()

    return WordEmbeddingsParams(
        task=args.task,
        corpus=args.corpus,
        model=args.model,
        size=args.size
    )


def get_word_embeddings_model_name(model: str = 'glove') -> str:
    model_name = {
        'glove': 'GloVe',
        'word2vec': 'Word2Vec',
        'fasttext': 'FastText'
    }
    return model_name[model]


def get_pretrained_word_embeddings_model_name(model: str = 'glove') -> list:
    model_name = {
        'glove': ['glove-wiki-gigaword-100', 'glove-wiki-gigaword-300'],
        'word2vec': ['word2vec-google-news-300'],
        'fasttext': ['fasttext-wiki-news-subwords-300']
    }
    return model_name[model]


def get_word_embeddings_filename(model: str = 'glove', size: int = 300) -> str:
    model_name = {
        'glove': f"glove-wiki-gigaword-{size}.txt",
        'word2vec': f"word2vec-google-news-{size}.txt",
        'fasttext': f"fasttext-wiki-news-subwords-{size}.txt"
    }
    return model_name[model]


def get_filtered_word_embeddings_filename(corpus: str = None, model: str = 'glove', size: int = 300) -> str:
    model_name = {
        'glove': f"glove-filtered-wiki-gigaword-{corpus}-{size}.txt",
        'word2vec': f"word2vec-filtered-google-news-{corpus}-{size}.txt",
        'fasttext': f"fasttext-filtered-wiki-news-subwords-{corpus}-{size}.txt",
    }
    return model_name[model]


def get_pretrained_word_embeddings_number_of_words(word_embeddings_path: str = '') -> str:
    fin = codecs.open(word_embeddings_path, 'r', encoding='utf-8')
    # The first line of the file contains the number of words in the vocabulary and the size of the vectors.
    n, d = map(int, fin.readline().split())
    return n


def filter_word_embeddings(source: str, dest: str, vocab: set) -> int:
    log.info(f"Filtering word embeddings from '{source}' to '{dest}'")

    tic = time.time()

    total = cnt = 0
    num_lines = get_pretrained_word_embeddings_number_of_words(source)
    with codecs.open(source, encoding="utf8") as fin:
        with codecs.open(dest, 'w', encoding='utf8') as fout:
            for line in tqdm(fin, total=num_lines, desc='lines'):
                total += 1
                # The first line of the file contains the number of words in the vocabulary and the size of the vectors.
                if total == 1:
                    continue
                word = line.split(' ', 1)[0]
                if word in vocab or word == '(' or word == ')':
                    cnt += 1
                    fout.write(line)

    log.info(f"Total word embeddings: {total}, after filtering: {cnt}.")
    log.info(f"Filtering word embeddings total time: {(time.time() - tic) / 60} minutes")
    return cnt


def get_embeddings(
        embed_path: str = '',
        embeddings_model: str = 'glove',
        embeddings_size: int = 300,
        pretrained: bool = True
) -> Tuple[np.ndarray, Dict]:
    word_embeds, w2i = [np.random.randn(embeddings_size)], {'_UNK_': 0}

    if pretrained or embeddings_model == 'glove':
        num_lines = sum(1 for _ in open(embed_path, 'r', encoding='utf8'))
    else:
        num_lines = get_pretrained_word_embeddings_number_of_words(embed_path)

    total = 0
    with codecs.open(embed_path) as f:
        for line in tqdm(f, total=num_lines, desc='lines'):
            total += 1
            # The first line of the file contains the number of words in the vocabulary and the size of the vectors.
            if not pretrained:
                if total == 1:
                    continue
            line = line.strip().split(' ')
            word, embed = line[0], line[1:]
            w2i[word] = len(word_embeds)
            word_embeds.append(np.array(embed, dtype=np.float32))

    # Left Round Bracket and Right Round Bracket
    if '(' in w2i:
        w2i['-LRB-'] = w2i['(']
    if ')' in w2i:
        w2i['-RRB-'] = w2i[')']

    # Start Of Sentence and End Of Sentence keys.
    w2i['_SOS_'] = len(word_embeds)
    word_embeds.append(np.random.randn(embeddings_size))
    w2i['_EOS_'] = len(word_embeds)
    word_embeds.append(np.random.randn(embeddings_size))

    return np.array(word_embeds, dtype=object), w2i
