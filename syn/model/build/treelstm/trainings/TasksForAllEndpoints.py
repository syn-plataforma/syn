#!/usr/bin/env python3
import os
import time
from pathlib import Path

import log4p
from dotenv import load_dotenv

from definitions import ROOT_DIR, SYN_ENV
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.model.ModelHelper import get_input_params
from syn.model.build.treelstm.trainings.tree_lstm import TreeLstmDuplicateTrain


########################################################################################################################
#
# $ nohup python3 -u -m syn.model.build.codebooks.creacion_de_tareas.TasksForAllEndpoints --c netBeans >> output.log &
#
########################################################################################################################


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()
    # Define el logger que se utilizará.
    logger = log4p.GetLogger(__name__)
    log = logger.logger

    log.info(f"INICIO DE LA EJECUCIÓN")
    env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
    load_dotenv(dotenv_path=env_path)

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Incializa las variables que almacenarán los argumentos de entrada.
    input_params = get_input_params()

    dup = TreeLstmDuplicateTrain(
        corpus='bugzilla',
        collection='clear',
        attention=False,
        attention_size=10,
        glove_size=100,
        hidden_size=100,
        max_input=200,
        batch_size=1,
        optimizer='ADAM',
        learning_rate=0.001,
        update_embeddings=True,
        patience=5
    ).load_or_run()

    output_dir = 'resultados/dump'
    # try:
    #     dup.delete_dynet(output_dir=output_dir)
    #     pass
    # except OSError:
    #     pass
    # # dup.run()
    # # classifier, result = dup.load_or_run()
    # classifier, results = dup.load_or_run_dynet()
    #
    # print({x: results[x] for x in ["Accuracy", "Precision", "Recall", "F1"]})

    # import matplotlib.pyplot as plt
    #
    # plt.plot(*results["roc"][:2])
    # plt.plot([0, 1], [0, 1], "k--")
    # plt.title("ROC")
    # plt.show()
    #
    # results["confusion"]


if __name__ == '__main__':
    main()
