#!/usr/bin/env python3
import os
import time
from pathlib import Path
from subprocess import check_output

import log4p
from dotenv import load_dotenv

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

    for project in os.environ["BUGZILLA_PROJECT_NAME"].split(","):
        for year in range(int(os.environ['BUGZILLA_FIRST_CREATION_YEAR']), int(os.environ['BUGZILLA_LAST_CREATION_YEAR'])):
            cmd = f"python -m syn.data.collect.bugzilla.create_bugzilla_mongodb_collection --p {project}" \
                  f" --db {os.environ['BUGZILLA_MONGODB_DATABASE_NAME']}" \
                  f" --c {project.lower()} --y {year} --sm 1 --em 12"

            # Ejecuta el comando en la consola Windows.
            check_output(cmd, shell=True)
            print(cmd)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutos")


if __name__ == '__main__':
    main()
