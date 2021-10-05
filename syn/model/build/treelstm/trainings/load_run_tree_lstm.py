import os
from pathlib import Path

from dotenv import load_dotenv

# Define a task and its parameters
from definitions import ROOT_DIR, SYN_ENV
from syn.model.build.codebooks.incidences import tasks

env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
load_dotenv(dotenv_path=env_path)
remote_base_path = os.environ["REMOTE_BASE_PATH"]


dataset = "eclipse"
glove_size = 100
max_input = 400
f"new_{dataset}_duplicate_det_task"
dup = tasks.TreeLstmDuplicateTrain(dataset, f"new_{dataset}_duplicate_det_task", attention=True, attention_size=50,
                                   glove_size=glove_size, hidden_size=300, max_input=max_input, batch_size=1,
                                   optimizer='ADAM',
                                   learning_rate=0.001,
                                   update_embeddings=False, patience=5)

try:
    # dup.delete(model_name='treelstm', remote_base_path=remote_base_path)
    pass
except OSError:
    pass
# dup.run()
# classifier, result = dup.load_or_run()
classifier, results = dup.load_or_run(model_name='treelstm', remote_base_path=remote_base_path)
# dup.delete(model_name='treelstm', remote_base_path=remote_base_path)
#

# dup.delete(model_name='treelstm', remote_base_path=remote_base_path)
# dup.delete2(model_name='')
print({x: results[x] for x in ["Accuracy", "Precision", "Recall", "F1"]})
