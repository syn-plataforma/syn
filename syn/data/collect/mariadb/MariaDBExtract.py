import os
from pathlib import Path

import log4p
from dotenv import load_dotenv
from sqlalchemy import create_engine

from definitions import ROOT_DIR, SYN_ENV

logger = log4p.GetLogger(__name__)
log = logger.logger

env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
load_dotenv(dotenv_path=env_path)

# Conexión a una BBDD MySQL utilizando un sqlachemy engine.
redmine_connection_string = 'mysql+mysqlconnector://' + os.environ.get('MYSQL_USER_REDMINE') \
                            + ':' + os.environ.get('MYSQL_PASSWORD_REDMINE') + '@' + \
                            os.environ['MYSQL_HOST_IP_REDMINE'] + ':' + os.environ['MYSQL_PORT_REDMINE'] + '/' \
                            + os.environ['MYSQL_DATABASE_REDMINE']

log.debug(redmine_connection_string)

redmine_sql_engine = create_engine(redmine_connection_string, echo=False)

# Conexión a una BBDD MySQL utilizando un sqlachemy engine.
redmine_syn_connection_string = 'mysql+mysqlconnector://' + os.environ.get('MYSQL_USER_REDMINE_SYN') + ':' \
                                + os.environ.get('MYSQL_PASSWORD_REDMINE_SYN') + '@' \
                                + os.environ['MYSQL_HOST_IP_REDMINE_SYN'] + ':' + os.environ['MYSQL_PORT_REDMINE_SYN'] \
                                + '/' + os.environ['MYSQL_DATABASE_REDMINE_SYN']

log.debug(redmine_syn_connection_string)

redmine_sql_syn_engine = create_engine(redmine_syn_connection_string, echo=False)
