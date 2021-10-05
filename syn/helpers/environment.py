from pathlib import Path

from dotenv import load_dotenv

from definitions import ROOT_DIR, SYN_ENV


def load_environment_variables():
    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)
