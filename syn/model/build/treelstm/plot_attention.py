import os
import time
from pathlib import Path

from tqdm import tqdm

from syn.helpers.hyperparams import get_input_params
from syn.helpers.logging import set_logger
from syn.helpers.nlp.embeddings import get_embeddings, get_filtered_word_embeddings_filename
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.treelstm.dataloader import DataLoader
from syn.model.build.common.task import Experiment
from syn.model.build.treelstm.dynetconfig import get_dynet
from syn.model.build.treelstm.dynetmodel import IssueClassifier

log = set_logger()
dy = get_dynet()

if __name__ == "__main__":
    log.info(f"Plotting attention ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()
    assert input_params is not None, f"No params provided."

    test = DataLoader(
        model=input_params['model']['architecture'],
        task=input_params['dataset']['task'],
        corpus=input_params['dataset']['corpus'],
        dataset_name='balanced_test' if input_params['dataset']['balance_data'] else 'test',
        query_limit=input_params['dataset']['query_limit'],
        save_dir=input_params['dataset']['dataset_save_dir']
    )

    # Word embeddings.
    embeddings_dir = Path(os.environ.get('DATA_PATH')) / 'word_embeddings'
    if input_params['model']['embeddings_pretrained'] or 'glove' == input_params['model']['embeddings_model']:
        embeddings_filename = \
            get_filtered_word_embeddings_filename(
                corpus=input_params['dataset']['corpus'],
                model=input_params['model']['embeddings_model'],
                size=input_params['model']['embeddings_size']
            )
    else:
        embeddings_filename = f"{input_params['model']['embeddings_model']}-{input_params['dataset']['corpus']}-" \
                              f"{input_params['model']['embeddings_size']}.txt"

    embeddings_path = Path(embeddings_dir) / input_params['model']['embeddings_model'] / embeddings_filename
    if not os.path.isfile(embeddings_path):
        log.error(f"No such filtered word embeddings file: '{embeddings_path}'.")
    assert os.path.isfile(embeddings_path), 'Ensure word embeddings file exists.'
    word_embed, w2i = get_embeddings(
        embeddings_path,
        input_params['model']['embeddings_model'],
        input_params['model']['embeddings_size'],
        input_params['model']['embeddings_pretrained']
    )

    classifier = Experiment(IssueClassifier, test, w2i, word_embed, input_params)

    for inst in tqdm(test, total=test.n_samples, desc='rows'):
        # build graph for this instance
        dy.renew_cg()

        # Document as Tuple(trees, attention_vectors).
        document = (inst[0], inst[1])

        _, _, attention_weights = classifier.model.predict(document)

        # Plot attention.
        classifier.model.issue_description_builder.attention.plot_attention(inst[0], attention_weights)

    log.info(f"Evaluating model total time = {((time.time() - initial_time) / 60)} minutes")
