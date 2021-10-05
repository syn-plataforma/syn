import os

# dynet_config object must be imported and used before import dynet.
import dynet_config

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger

load_environment_variables()
log = set_logger()


def get_dynet():
    if bool(os.environ['DYNET_USE_GPU']):
        dynet_config.set_gpu()

    dynet_config.set(
        mem=os.environ['DYNET_MEM'],
        random_seed=int(os.environ['DYNET_SEED']),
        autobatch=int(os.environ['DYNET_AUTOBATCH']),
        requested_gpus=int(os.environ['DYNET_GPUS'])
    )

    log.info(f"DyNet config: {dynet_config.get()}")

    import dynet as dy
    return dy
