import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Tuple

import psutil

from definitions import ROOT_DIR
from syn.helpers.logging import set_logger

# Defines logger.
log = set_logger()


def check_process_running_by_name(process_name):
    """
    Check if there is any running process that contains the given name processName.
    """
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if process_name.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def check_process_running_by_file_path(file_path: Path = None) -> bool:
    """
    Check if there is any running process that contains the given name process file path.
    """
    # Find Python processes.
    python_process = find_process_id_by_name('python')

    # Build module path as class path.
    file_path = file_path
    module_file_path = file_path.relative_to(Path(ROOT_DIR))
    module_file_path = str(module_file_path).replace(os.sep, '.').replace('.py', '')

    # Iterate over all python running process to check if there is another process running with the same file path.
    is_already_running = False
    num_running_processes = 0
    for proc in python_process:
        try:
            # Check if process contains the given process file path string.
            if "windows" != str(platform.system()).lower():
                if module_file_path in proc['cmdline']:
                    num_running_processes += 1
            else:
                if module_file_path in proc['cmdline'] or str(file_path.as_posix()) in proc['cmdline']:
                    if sys.executable in proc['cmdline'] or 'python' in proc['cmdline']:
                        num_running_processes += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if num_running_processes > 1:
        is_already_running = True

    return is_already_running


def find_process_id_by_name(process_name):
    """
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    """
    process_objects = []
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            p_info = proc.as_dict(attrs=['pid', 'name', 'create_time', 'cmdline'])
            # Check if process name contains the given name string.
            if process_name.lower() in p_info['name'].lower():
                process_objects.append(p_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_objects


def check_same_python_module_already_running(file_info: Tuple) -> None:
    # Process file path.
    module_file_path = Path(file_info[0]) / file_info[1]

    log.info(f"Running module path: '{str(module_file_path)}'")
    if check_process_running_by_file_path(module_file_path):
        log.error(f"Module: '{str(module_file_path)}' is already running.")
        raise SystemExit(0)


def get_env_var_path_sep(os_name):
    # os.name: The name of the operating system dependent module imported.The following names have currently been
    # registered: 'posix', 'nt', 'java'.
    env_var_path_sep = {
        "posix": ":",
        "nt": ";",
        "java": ""
    }
    return env_var_path_sep[os_name]


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def get_java_classpath():
    return get_env_var_path_sep(os.name).join([
        str(Path(ROOT_DIR) / 'lib' / 'stanford' / '*'),
        str(Path(ROOT_DIR) / 'lib' / 'mongodb' / '*'),
        str(Path(ROOT_DIR) / 'lib' / 'syn' / '*')
    ])
