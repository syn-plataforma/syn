#!/usr/bin/env python3
import os
import time

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_command(
        os_name: str = 'posix',
        project: str = 'eclipse',
        year: int = '2001'
):
    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    arguments = f" -m syn.data.collect.gerrit.create_gerrit_mongodb_collection --collection_name {project} --year {year}"
    cmd = {
        'posix': f"python3 {arguments}",
        'nt': f"python {arguments}",
        'java': ''
    }

    return cmd[os_name]


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    for project in os.environ["GERRIT_PROJECT_NAME"].split(","):
        for year in range(int(os.environ['GERRIT_FIRST_CREATION_YEAR']), int(os.environ['GERRIT_LAST_CREATION_YEAR'])):
            tic = time.time()
            log.info(f"Retrieving Gerrit issues for year '{year}' ...")
            cmd = get_command(os.name, project, year)

            # Run command.
            log.info(f"Running command: '{cmd}'")
            os.system(cmd)
            log.info(f"Retrieving Gerrit issues execution time = {((time.time() - tic) / 60)} minutes")

    final_time = time.time()
    log.info(f"Retrieving Gerrit issues total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
