import argparse
import os
import time
import datetime
import xmlrpc
from pathlib import Path

import log4p
import requests
from dotenv import load_dotenv

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.bugzilla.BugzillaAPIParams import BugzillaAPIParams
from syn.helpers.mongodb import MongoDBParams

# Stores the execution start time to calculate the time it takes for the module to execute.
initial_time = time.time()
# Define el logger que se utilizará.
logger = log4p.GetLogger(__name__)
log = logger.logger


def get_input_params():
    parser = argparse.ArgumentParser(
        description='Recupera incidencias de Bugzilla y las almacena en una colección MongoDB.'
    )
    # project
    parser.add_argument(
        "--p",
        default="eclipse",
        type=str,
        choices=['eclipse', 'netbeans', 'openofice', 'mozilla', 'gcc'],
        help="Proyecto para el que se quiere recuperar incidencias."
    )
    # year
    parser.add_argument(
        "--y",
        default=2001,
        type=int,
        help="Año para el que se quiere recuperar incidencias."
    )
    # start_month
    parser.add_argument(
        "--sm",
        default=1,
        type=int,
        help="Número del mes a partir del que se quiere recuperar incidencias."
    )
    # end_month
    parser.add_argument(
        "--em",
        default=12,
        type=int,
        help="Número del mes hasta el que se quiere recuperar incidencias."
    )
    # query_limit
    parser.add_argument(
        "--ql",
        default=0,
        type=int,
        help="Número máximo de incidencias a recuperar (0 sin límite)."
    )
    # include_fields
    parser.add_argument(
        "--incf",
        default=None,
        type=str,
        help="Lista con el nombre de los campos separados por comas que se quieren incluir en los resultados."
    )
    # get_comments
    parser.add_argument(
        "--gc",
        default=True,
        action='store_true',
        help="Establece si se deben recuperar los comentarios asociados a cada incidencia."
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
        "--db",
        default="test",
        type=str,
        help="Nombre de la base de datos en la que se almacenarán las incidencias."
    )
    # mongodb_collection_name
    parser.add_argument(
        "--c",
        default="test",
        type=str,
        help="Nombre de la colección en la que se almacenarán las incidencias."
    )
    # drop mongodb_collection_name
    parser.add_argument(
        "--dc",
        default=True,
        action='store_true',
        help="Elimina la colección en la que se almacenarán las incidencias antes de escribir en ella."
    )

    args = parser.parse_args()
    return {
        "bz_api_params": BugzillaAPIParams(
            project=args.p,
            year=args.y,
            start_month=args.sm,
            end_month=args.em,
            query_limit=args.ql,
            include_fields=args.incf,
            get_comments=args.gc
        ),
        "mongo_params": MongoDBParams(
            host=args.mh,
            port=args.mp,
            db_name=args.db,
            collection_name=args.c,
            drop_collection=args.dc
        )
    }


def create_query_by_date(min_creation_ts, max_creation_ts, max_results, include_fields=None):
    if include_fields is None:
        include_fields = "_all,_extra"
    return f"?query_format=advanced&include_fields={include_fields}" \
           f"&f1=creation_ts&o1=greaterthaneq&v1={min_creation_ts}" \
           f"&f2=creation_ts&o2=lessthan&v2={max_creation_ts}" \
           f"&limit={max_results}&order=id"


def create_xml_rpc_query_by_date(min_creation_ts, max_creation_ts, max_results, include_fields=None):
    query = f"?query_format=advanced"

    if include_fields is not None:
        for field in include_fields:
            query += f"&include_fields=[{include_fields},"

        # Elimina la última ",".
        query = query[:-1]
        query += "]"

    if min_creation_ts is not None:
        query += f"&f1=creation_ts&o1=greaterthaneq&v1={min_creation_ts}"

    if max_creation_ts is not None:
        query += f"&f2=creation_ts&o2=lessthan&v2={max_creation_ts}"

    if max_results is not None:
        query += f"&limit={max_results}"

    query += "& order = id"

    return query


def create_xml_rpc_query_by_date0(min_creation_ts, max_creation_ts, max_results, include_fields=None):
    query = {"query_format": "advanced"}
    if include_fields is not None:
        query["include_fields"] = include_fields

    if min_creation_ts is not None:
        query["f1"] = "creation_ts"
        query["o1"] = "greaterthaneq"
        query["v1"] = min_creation_ts

    if max_creation_ts is not None:
        query["f2"] = min_creation_ts
        query["o2"] = "lessthan"
        query["v2"] = min_creation_ts

    if max_results is not None:
        query["limit"] = max_creation_ts

    return query


def add_comments(bz_api_url, bugzilla_bugs):
    json_data = list()
    for bug in bugzilla_bugs:
        bug_json_response = requests.get(f"{bz_api_url}/bugs/rest/bug/{bug['id']}/comment").json()
        comments = bug_json_response["bugs"][f"{bug['id']}"]["comments"]
        bug["comments"] = comments
        json_data.append(bug)

    return json_data


def add_description(bz_api_url, bugzilla_bugs):
    json_data = list()
    for bug in bugzilla_bugs:
        bug_json_response = requests.get(f"{bz_api_url}/bugs/rest/bug/{bug['id']}/comment").json()
        description = bug_json_response["bugs"][f"{bug['id']}"]["comments"][0]["text"]
        bug["description"] = description
        json_data.append(bug)

    return json_data


def call_bugzilla_bugs_rest_api(bz_api_url, min_creation_ts, max_creation_ts, max_results=0,
                                include_fields=None, get_comments=True):
    # Construye la query que se va a enviar en la llamada a la API de Bugzilla.
    query = create_query_by_date(min_creation_ts, max_creation_ts, max_results, include_fields)
    log.info(query)

    get_bz_bugs_start_time = time.time()
    # Obtiene las incidencias utlizando la API de Bugzilla.
    bugs_json_response = requests.get(f"{bz_api_url}/bugs/rest/bug{query}").json()["bugs"]
    get_bz_bugs_end_time = time.time()
    log.info(
        f"Tiempo empleado en ejecutar la consulta a la API de Bugzilla para recuperar las incidencias "
        f"con fecha de creación mayor o igual que {min_creation_ts} y menor que {max_creation_ts} = "
        f"{((get_bz_bugs_end_time - get_bz_bugs_start_time) / 60)} minutos"
    )
    log.info(
        f"Número de incidencias con fecha de creación mayor o igual que {min_creation_ts} "
        f"y menor que {max_creation_ts} = {len(bugs_json_response)}"
    )

    # Comprueba si se quieren recuperar los comentarios.
    if get_comments:
        get_bz_bugs_comments_start_time = time.time()

        # Añade los comentarios consultando la API de Bugzilla para cada una de las incidencias
        bugs_with_description = add_comments(bz_api_url, bugs_json_response)

        get_bz_bugs_comments_end_time = time.time()
        log.info(
            f"Tiempo empleado en ejecutar la consulta a la API de Bugzilla para recuperar los comentarios "
            f"de las incidencias = "
            f"{((get_bz_bugs_comments_end_time - get_bz_bugs_comments_start_time) / 60)} minutos"
        )

        return bugs_with_description

    return bugs_json_response


def xmlrpc_client_datetime_to_datetime(field):
    if isinstance(field, xmlrpc.client.DateTime):
        field = datetime.datetime.strptime(str(field), '%Y%m%dT%H:%M:%S')

    return field


def bzapi_bug_to_dict(bzapi_bug):
    result = {}
    for bug_field in bzapi_bug.__dict__['_bug_fields']:
        result[bug_field] = xmlrpc_client_datetime_to_datetime(bzapi_bug.__dict__[bug_field])

    return result


def call_bugzilla_bugs_xmlrpc_search(bz_api_url, min_creation_ts, max_creation_ts, max_results=0,
                                     include_fields=None, get_comments=True):
    from scripts import bugzilla

    # TODO: Corregir el error con include_fields "TypeError: Bug object needs a bug_id"
    include_fields = ["id", "bug_id"]
    # Construye la query que se va a enviar en la llamada a la API de Bugzilla.
    query = create_xml_rpc_query_by_date(min_creation_ts, max_creation_ts, max_results, None)
    log.info(query)

    bzapi = bugzilla.Bugzilla(bz_api_url, cookiefile=None)
    bzapi_query = bzapi.url_to_query(f"{os.environ['BUGZILLA_GCC_XML_RPC_SEARCH_URL']}{query}")

    get_bz_bugs_start_time = time.time()
    # Obtiene las incidencias utlizando la API de Bugzilla.
    bugs = bzapi.query(bzapi_query)

    get_bz_bugs_end_time = time.time()
    log.info(
        f"Tiempo empleado en ejecutar la consulta a la API de Bugzilla para recuperar las incidencias "
        f"con fecha de creación mayor o igual que {min_creation_ts} y menor que {max_creation_ts} = "
        f"{((get_bz_bugs_end_time - get_bz_bugs_start_time) / 60)} minutos"
    )
    log.info(
        f"Número de incidencias con fecha de creación mayor o igual que {min_creation_ts} "
        f"y menor que {max_creation_ts} = {len(bugs)}"
    )

    get_bz_bugs_comments_start_time = time.time()
    bugs_dict_response = []
    for bug in bugs:
        bug_dict = bzapi_bug_to_dict(bug)
        bug_dict['comments'] = []
        # Comprueba si se quieren recuperar los comentarios.
        get_bz_bugs_comments_start_time = time.time()
        if get_comments:
            # Añade los comentarios consultando la API de Bugzilla para cada una de las incidencias
            comments = bug.getcomments()
            # bugs_dict_response['comments'] = []
            for comment in comments:
                for key in comment:
                    comment[key] = xmlrpc_client_datetime_to_datetime(comment[key])

            bug_dict['comments'] = comments

        bugs_dict_response.append(bug_dict)

    get_bz_bugs_comments_end_time = time.time()
    log.info(
        f"Tiempo empleado en ejecutar la consulta a la API de Bugzilla para recuperar los comentarios "
        f"de las incidencias = "
        f"{((get_bz_bugs_comments_end_time - get_bz_bugs_comments_start_time) / 60)} minutos"
    )

    return bugs_dict_response


def call_issue_tracking_api_factory(
        project="eclipse",
        min_creation_ts="2001-01-01",
        max_creation_ts="2001-02-01",
        max_results=10,
        include_fields=None,
        get_comments=True
):
    # Carga el fichero de configuración para el entorno.
    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    api_call = {
        "eclipse": {"url": os.environ["BUGZILLA_ECLIPSE_REST_API_URL"], "function": call_bugzilla_bugs_rest_api},
        "netbeans": {"url": os.environ["BUGZILLA_NETBEANS_XML_RPC_API_URL"],
                     "function": call_bugzilla_bugs_xmlrpc_search},
        "openofice": {"url":
                          os.environ["BUGZILLA_ECLIPSE_REST_API_URL"], "function": call_bugzilla_bugs_xmlrpc_search},
        "mozilla": {"url": os.environ["BUGZILLA_GCC_REST_API_URL"], "function": call_bugzilla_bugs_rest_api},
        "gcc": {"url": os.environ["BUGZILLA_GCC_XML_RPC_API_URL"], "function": call_bugzilla_bugs_xmlrpc_search}
        # "spark": {"url": os.environ["JIRA_APACHE_SERVER"], "function": call_jira_issues_rest_api_search}
    }
    # project = 'netbeans'
    return api_call[project]["function"](
        api_call[project]["url"],
        min_creation_ts,
        max_creation_ts,
        max_results,
        include_fields,
        get_comments
    )


def get_bugzilla_bugs_by_date_range(
        project="eclipse",
        min_creation_ts="2001-01-01",
        max_creation_ts="2001-02-01",
        max_results=10,
        include_fields=None,
        get_comments=True

):
    factory_obj = None
    try:
        factory_obj = call_issue_tracking_api_factory(project, min_creation_ts, max_creation_ts, max_results,
                                                      include_fields, get_comments)
    except ValueError as e:
        log.error(e)
    return factory_obj
