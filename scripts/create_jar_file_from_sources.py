import os
from pathlib import Path

from definitions import ROOT_DIR
from syn.helpers.logging import set_logger
from syn.helpers.system import get_env_var_path_sep

# Defines logger.
log = set_logger()

if __name__ == '__main__':
    # Java CLASSPATH.
    lib_dir = Path(ROOT_DIR) / 'lib'
    classpath = get_env_var_path_sep(os.name).join([
        str(Path(ROOT_DIR) / 'lib' / 'stanford' / '*'),
        str(Path(ROOT_DIR) / 'lib' / 'mongodb' / '*')
    ])

    # Defines output directory for compiled java sources.
    class_dir = Path(ROOT_DIR) / 'lib' / 'class'

    # Defines javac executable.
    javac_exe = Path(os.environ['JAVA_HOME']) / 'bin' / 'javac.exe'

    # Defines java sources files directory.
    java_files_dir = Path(ROOT_DIR) / 'syn' / 'model' / 'build' / 'treelstm' / 'vectorizer' / 'java' / 'src' / '*.java'

    # Command for compile java sources.
    javac_cmd = f"{javac_exe} -Xlint:unchecked -encoding utf8 -classpath {classpath} -d {class_dir} {java_files_dir}"

    log.info(f"Command to execute: '{javac_cmd}'.")
    os.system(javac_cmd)

    # Defines jar executable.
    jar_exe = Path(os.environ['JAVA_HOME']) / 'bin' / 'jar.exe'
    jar_dir = Path(ROOT_DIR) / 'lib' / 'syn' / 'syn.jar'
    jar_cmd = f"{jar_exe} cvf \"{jar_dir}\" -C \"{class_dir}\" ."

    log.info(f"Command to execute: '{jar_cmd}'.")

    os.system(jar_cmd)
