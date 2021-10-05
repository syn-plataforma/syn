import re
import string
from unicodedata import normalize

from nltk import SnowballStemmer
from nltk.corpus import stopwords


def clean_doc_replace(doc):
    doc = doc.replace('--', ' ')
    return doc


def clean_doc_split(doc):
    tokens = "".join((char if char.isalpha() else " ") for char in str(doc)).split()
    #     tokens= "".join(doc).split()
    return tokens


def clean_doc_punctuaction(tokens):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    return tokens


def clean_doc_isalpha(tokens):
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


def clean_doc_lower(tokens):
    tokens = [word.lower() for word in tokens]
    return tokens


def clean_doc_trim(tokens):
    tokens = [word.strip() for word in tokens]
    return tokens


def clean_doc_stopW(tokens):
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return tokens


def clean_doc_lenword(tokens):
    len_word = 3
    tokens = [w for w in tokens if not len(w) < len_word]
    return tokens


def clean_doc_diacri(tokens):
    tokens = [normalize('NFC', token) for token in tokens]
    return tokens


def clean_doc_lem(tokens):
    lemmatizer_simple = SnowballStemmer(language='english')
    stems = [lemmatizer_simple.stem(token) for token in tokens]
    return stems


def clean_doc_vector(embedding_matrix, tokens):
    tokens = [embedding_matrix[w] for w in tokens]
    return tokens
