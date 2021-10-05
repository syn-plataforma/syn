import argparse

from syn.helpers.mongodb import MongoDBFilterParams, MongoDBParams
from syn.helpers.nlp.NLPParams import NLPParams


def get_input_params():
    parser = argparse.ArgumentParser(
        description='Lee un fichero de texto y almacena su contenido en MongoDB.'
    )
    # java_class_name
    parser.add_argument(
        "--jcn", "--java-class-name",
        default="UpdateMongoDBNLPFields",
        type=str,
        choices=['UpdateMongoDBNLPFields', 'UpdateMongoDBTokenField', 'UpdateMongoDBScoreField'],
        help="Nombre de la colección MongoDB que contiene los campos de texto a normalizar."
    )
    # max-num-tokens
    parser.add_argument(
        "--mnt", "--max-num-tokens",
        default=150,
        type=int,
        help="Número máximo de tokens que deberá tener la sentencia para procesarla."
    )
    # mongodb-host
    parser.add_argument(
        "--mh", "--mongodb-host",
        default="localhost",
        type=str,
        choices=['localhost', 'syn.altgovrd.com'],
        help="Nombre del servidor MongoDB."
    )
    # mongodb-port
    parser.add_argument(
        "--mp", "--mongodb-port",
        default=30017,
        type=int,
        help="Puerto del servidor MongoDB."
    )
    # mongodb-db-name
    parser.add_argument(
        "--db", "--db-name",
        default="bugzilla",
        type=str,
        choices=['bugzilla', 'eclipse', 'netBeans', 'openOffice'],
        help="Nombre de la colección MongoDB que contiene los campos de texto a normalizar."
    )
    # mongodb-collection-name
    parser.add_argument(
        "--c", "--collection-name",
        default="normalized_clear",
        type=str,
        choices=['clear_embeddings', 'normalized_clear_embeddings', 'clear', 'initial', 'normalized_clear'],
        help="Nombre de la colección en la que se almacenarán las incidencias."
    )
    # column-name
    parser.add_argument(
        "--cl", "--column-name",
        default="normalized_description",
        type=str,
        help="Nombre de la columna que contiene el texto para la que se van a obtener los embeddings."
    )
    # start-year
    parser.add_argument(
        "--sy", "--start-year",
        default=2000,
        type=int,
        help="Año a partir del cual se recuperarán documentos de la colección MongoDB."
    )
    # end-year
    parser.add_argument(
        "--ey", "--end-year",
        default=2021,
        type=int,
        help="Año dhasta el que se recuperarán documentos de la colección MongoDB."
    )
    # parser-model
    parser.add_argument(
        "--pm", "--parser-model",
        default="corenlp",
        type=str,
        choices=['corenlp', 'srparser'],
        help="Modelo del parser de Stanford utilizado."
    )
    # get-trees
    parser.add_argument(
        '--get-trees',
        default=True,
        dest='get-trees',
        action='store_true',
        help="Obtener árboles sintácticos."
    )
    # no-get-trees
    parser.add_argument(
        '--no-get-trees',
        dest='get-trees',
        action='store_false',
        help="No obtener árboles sintácticos."
    )
    # get-embeddings
    parser.add_argument(
        '--get-embeddings',
        default=True,
        dest='get-embeddings',
        action='store_true',
        help="Obtener los embeddings para las hojas de los árboles sintácticos."
    )
    # no-get-embeddings
    parser.add_argument(
        '--no-get-embeddingss',
        dest='get-embeddings',
        action='store_false',
        help="No obtener los embeddings para las hojas de los árboles sintácticos."
    )
    # get-coherence
    parser.add_argument(
        '--get-coherence',
        default=True,
        dest='get-embeddings',
        action='store_true',
        help="Obtener la coherencia de los árboles sintácticos."
    )
    # no-get-coherence
    parser.add_argument(
        '--no-get-coherence',
        dest='get-coherence',
        action='store_false',
        help="No obtener la coherencia de los árboles sintácticos."
    )

    args = parser.parse_args()

    return {
        "filter_params": MongoDBFilterParams(
            column_name=args.cl,
            start_year=args.sy,
            end_year=args.ey
        ),
        "mongo_params": MongoDBParams(
            host=args.mh,
            port=args.mp,
            db_name=args.db,
            collection_name=args.c
        ),
        "nlp_params": NLPParams(
            java_class_name=args.jcn,
            max_num_tokens=args.mnt,
            parser_model=args.pm,
            get_trees=args.__getattribute__('get-trees'),
            get_embeddings=args.__getattribute__('get-embeddings'),
            get_coherence=args.__getattribute__('get-coherence')
        )
    }
