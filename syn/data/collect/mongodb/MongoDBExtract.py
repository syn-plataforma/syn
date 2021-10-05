import os
from pathlib import Path

import log4p
from dotenv import load_dotenv
from sqlalchemy import create_engine

from definitions import ROOT_DIR, SYN_ENV
from pymongo import MongoClient

##Conexión a la base de datos de mongodb
client = MongoClient()

logger = log4p.GetLogger(__name__)
log = logger.logger

env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
load_dotenv(dotenv_path=env_path)

mongodb_connection_string = MongoClient(host=os.environ['MONGO_HOST_IP'], port=int(os.environ['MONGO_PORT']), username=os.environ['MONGO_USERNAME'], password=os.environ['MONGO_PASSWORD'])



