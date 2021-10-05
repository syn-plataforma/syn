import os
from pathlib import Path

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger

load_environment_variables()
log = set_logger()


def get_command(
        os_name: str = 'posix',
        class_path: str = '',
        port: int = 9000,
        timeout: int = 15000
):
    # Defines arguments.
    arguments = f"-mx4g -cp \"{class_path}/*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer " \
                f"-port {port} -timeout {timeout}"

    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    cmd = {
        'posix': f"{str(Path(os.environ['JAVA_HOME']))} {arguments} &"
        if os.environ['JAVA_HOME'] != '' else f"java {arguments}",
        'nt': f"{str(Path(os.environ['JAVA_HOME']) / 'bin' / 'java.exe')} {arguments}"
        if os.environ['JAVA_HOME'] != '' else f"java {arguments}",
        'java': ''
    }

    return cmd[os_name]


if __name__ == '__main__':
    log.info('Starting Stanford CoreNLP server ...')

    start_cmd = get_command(
        os.name, os.environ['CORENLP_HOME'],
        int(os.environ['CORENLP_SERVER_PORT']),
        int(os.environ['CORENLP_SERVER_TIMEOUT'])
    )

    # Run command.
    log.info(f"Running command: '{start_cmd}'")
    os.system(start_cmd)
