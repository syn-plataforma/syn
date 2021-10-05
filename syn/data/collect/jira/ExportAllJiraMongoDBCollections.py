#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv
from subprocess import check_output

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.system import check_same_python_module_already_running


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()
    # Define el logger que se utilizará.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    for project in os.environ["JIRA_PROJECT_NAME"].split(","):
        for year in range(2001, 2021):
            cmd = f"mongoexport /host {os.environ['MONGO_HOST_IP']} /port {os.environ['MONGO_PORT']}" \
                  f" /username {os.environ['MONGO_USERNAME']} /password {os.environ['MONGO_PASSWORD']}" \
                  f" /authenticationDatabase admin /authenticationMechanism SCRAM-SHA-1" \
                  f" /db {os.environ['JIRA_MONGODB_DATABASE_NAME']}" \
                  f" /collection {project.lower()}_{year}_{year + 1}" \
                  f" /out {Path(ROOT_DIR) / 'data' / project.lower()}_{year}_{year + 1}.json"

            # Ejecuta el comando en la consola Windows.
            check_output(cmd, shell=True)
            print(cmd)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
