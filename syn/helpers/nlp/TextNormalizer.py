import argparse
import re
import unicodedata

import log4p
import nltk.data
from nltk.tokenize.toktok import ToktokTokenizer

from syn.helpers.mongodb import MongoDBParams
from syn.helpers.nlp.Contractions import CONTRACTION_MAP
from syn.helpers.nlp.TextNormalizerAPIParams import TextNormalizerAPIParams

# Define el logger que se utilizará.
logger = log4p.GetLogger(__name__)
log = logger.logger


def get_input_params():
    parser = argparse.ArgumentParser(
        description="Normaliza el texto de los campos que se pasen como parámetro en una lista y crea una nueva "
                    "colección MongoDB que conserva los campos originales y añade los nuevos campos normalizados."
    )
    # env
    parser.add_argument(
        "--e", "--environmet",
        default="aws",
        type=str,
        choices=['local', 'aws'],
        help="Entorno del que se leerá o en el se escribirá la colección MongoDB que contiene los campos de texto "
             "a normalizar."
    )
    # batches
    parser.add_argument(
        "--b", "--batches",
        default=1,
        type=int,
        help="Número de trozos en el que se divide el dataframe para guardarlo en MongoDB en varias etapas."
    )
    # output_format
    parser.add_argument(
        "--of", "--output_format",
        default="mongodb",
        type=str,
        choices=['mongodb', 'csv'],
        help="Formato del fichero de salida."
    )
    # columns_names
    parser.add_argument(
        "--cn", "--columns_names",
        nargs='*',
        default=['short_desc', 'description'],
        type=str,
        # choices=['short_desc', 'description'],
        help="Nombre de las columnas que se quieren normalizar, como por ejemplo 'short_desc' y 'description'"
    )
    # drop_original_columns
    parser.add_argument(
        "--doc", "--drop_original_columns",
        default=False,
        # action='store_true',
        help="Elimina las columnas originales en la colección MongoDB de salida."
    )
    # split_new_line
    parser.add_argument(
        "--snl", "--split_new_line",
        default=False,
        # action='store_true',
        help="Transforma los saltos de línea en '.'."
    )
    # to_lower_case
    parser.add_argument(
        "--tlc", "--to_lower_case",
        default=True,
        action='store_true',
        help="Transforma los tokens a minúsculas."
    )
    # mongodb_host
    parser.add_argument(
        "--mh", "--mongodb_host",
        default="localhost",
        type=str,
        choices=['localhost', 'syn.altgovrd.com'],
        help="Nombre del servidor MongoDB."
    )
    # mongodb_port
    parser.add_argument(
        "--mp", "--mongodb_port",
        default=30017,
        type=int,
        help="Puerto del servidor MongoDB."
    )
    # mongodb_db_name
    parser.add_argument(
        "--db", "--db_name",
        default="eclipse",
        type=str,
        choices=['eclipse', 'netBeans', 'openOffice'],
        help="Nombre de la colección MongoDB que contiene los campos de texto a normalizar."
    )
    # mongodb_collection_name
    parser.add_argument(
        "--c", "--collection_name",
        default="clear",
        type=str,
        choices=['clear', 'initial'],
        help="Nombre de la colección en la que se almacenarán las incidencias."
    )

    args = parser.parse_args()
    return {
        "nlp_api_params": TextNormalizerAPIParams(
            env=args.e,
            batches=args.b,
            output_format=args.of,
            columns_names=args.cn,
            drop_original_columns=args.doc,
            split_new_line=args.snl,
            to_lower_case=args.tlc
        ),
        "mongo_params": MongoDBParams(
            host=args.mh,
            port=args.mp,
            db_name=args.db,
            collection_name=args.c
        )
    }


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = ToktokTokenizer()


def get_len(text):
    if isinstance(text, list):
        return len(text)
    elif isinstance(text, str):
        return len(str(text.strip()))


def normalize_empty_list(text):
    if get_len(text) > 0:
        return text
    else:
        log.debug(f"=====> {text}")
        return "-"


def list_to_string(lst):
    # initialize an empty string
    result = ""

    # traverse in the string
    for elem in lst:
        result += elem

        # return string
    return result


def object_to_string(obj):
    # Comprueba el tipo del texto de entrada.
    if isinstance(obj, type(None)):
        return ""
    elif isinstance(obj, list):
        return list_to_string(obj)
    elif isinstance(obj, float):
        return str(obj) if obj is not None else ""

    return obj


def remove_whitespaces(text):
    return ' '.join(text.split())


def remove_accented_chars(text):
    if isinstance(text, list):
        log.error(text)
    try:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    except ValueError as err:
        print('Handling run-time error:', err)
        log.error(text)
    return text


def remove_repeated_characters_in_token(token, characters=". "):
    # New created word index.
    index = 0

    # Iter over word characters.
    for i in range(0, len(token)):
        # Check if token[i] appears before.
        for j in range(0, i + 1):
            if token[i] in characters and token[i] == token[j]:
                break

            # Add character to result.
            if j == i:
                token[index] = token[i]
                index += 1

    return "".join(token[:index])


def count_tokens(tokenized_sentences: list = None) -> int:
    count = 0
    for tokenized_sentence in tokenized_sentences:
        count += len(tokenized_sentence)

    return count


def get_codebooks_tokens(text: str = None) -> list:
    tokens = tokenizer.tokenize(text)
    return [[token.strip() for token in tokens]]


def remove_repeated_characters(text, to_lower_case=False, characters="."):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if to_lower_case:
        filtered_tokens = [remove_repeated_characters_in_token(list(token.lower()), characters) for token in tokens]
    else:
        filtered_tokens = [remove_repeated_characters_in_token(list(token), characters) for token in tokens]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9.\s]' if not remove_digits else r'[^a-zA-Z.\s]'
    text = re.sub(pattern, '', text)
    return text


def normalize_sentence(sentence, to_lower_case):
    line = remove_whitespaces(sentence)
    line = remove_accented_chars(line)
    line = remove_repeated_characters(line, to_lower_case)
    line = expand_contractions(line, CONTRACTION_MAP)
    line = remove_special_characters(line, remove_digits=False)
    return line


def normalize_incidence(text, split_new_line=False, to_lower_case=False):
    result = ""
    if split_new_line:
        sentences = []
        for paragraph in object_to_string(text).split('\n'):
            sentences.extend(sent_detector.tokenize(paragraph))
        tokenized_text = '.'.join(sentences)
    else:
        tokenized_text = text

    try:
        result = normalize_sentence(object_to_string(tokenized_text), to_lower_case)
    except ValueError as err:
        log.error(f"[RUN TIME ERROR] {err}\n {text}")
    return result


def normalize_df_column(df, column, input_params):
    df[f"normalized_{column}"] = df[column].progress_apply((
        lambda x: normalize_incidence(x, input_params['nlp_api_params'].split_new_line,
                                      input_params['nlp_api_params'].to_lower_case))).copy()

    return df
