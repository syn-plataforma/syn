import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Define a task and its parameters
from definitions import ROOT_DIR, SYN_ENV
from syn.model.build.codebooks.incidences import tasks

env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
load_dotenv(dotenv_path=env_path)
remote_base_path = os.environ["REMOTE_BASE_PATH"]

dataset = "eclipse"
column = 'severity'
glove_size = 300
max_input = 200
initial_time = time.time()
# Poner base de datos como variable de entorno.

dup = tasks.TreeLstmCategoricalTrain(dataset, f"new_{dataset}_{column}_task", attention=True, attention_size=50,
                                     glove_size=glove_size, hidden_size=100, max_input=max_input, batch_size=1,
                                     optimizer='ADAM',
                                     learning_rate=0.001,
                                     update_embeddings=True, patience=5, column='bug_' + column, num_samples=-1,
                                     train_porcent=0.8,
                                     balanced=True, num_cat=5)

try:
    # dup.delete(model_name='treelstm', remote_base_path=remote_base_path)
    pass
except OSError:
    pass
# dup.run()
classifier, result = dup.load_or_run(model_name='treelstm', remote_base_path=remote_base_path)
# dup.delete(model_name='treelstm', remote_base_path=remote_base_path)

# classifier, result = dup.run_and_pickle(model_name='treelstm', remote_base_path=remote_base_path)

# dup.delete(model_name='treelstm', remote_base_path=remote_base_path)
# dup.delete2(model_name='')
# print({x: result[x] for x in ["confusion", "other"]})
print(result['confusion'])
# conf = result["confusion"]
# import pandas as pd
#
# df_cm = pd.DataFrame(conf, index=[i for i in ["blocker", "critical", "major", "minor", "trivial"]],
#                      columns=[i for i in ["blocker", "critical", "major", "minor", "trivial"]])
# import seaborn as sn
#
# heat = sn.heatmap(df_cm, annot=True, fmt="d")
# heat.get_figure().savefig('.prueba_matrix.png')
# heat.get_figure().clf()

print(result['other'])
# dup._db_store()
print("Tiempo ejecución: ", (time.time() - initial_time) / 60)
