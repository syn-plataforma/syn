import os
import time
from pathlib import Path

from syn.helpers.hyperparams import get_codebooks_input_params
from syn.helpers.logging import set_logger
from syn.helpers.nlp.embeddings import get_embeddings, get_filtered_word_embeddings_filename
from syn.helpers.system import check_same_python_module_already_running
from syn.helpers.task import get_tokens_column_name, get_structured_data_column_name
from syn.helpers.codebooks.dataloader import DataLoader
from syn.model.build.codebooks.codebooks_task import Train

log = set_logger()

if __name__ == "__main__":
    log.info(f"Training model ...")
    initial_time = time.time()
    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load the parameters.
    input_params = get_codebooks_input_params()

    train = DataLoader(
        model=input_params['model']['architecture'],
        task=input_params['dataset']['task'],
        corpus=input_params['dataset']['corpus'],
        dataset_name='balanced_train' if input_params['dataset']['balance_data'] else 'train',
        tokens_columns=get_tokens_column_name(
            input_params['dataset']['task'], input_params['dataset']['corpus']
        ),
        structured_data_columns=get_structured_data_column_name(
            input_params['dataset']['task'], input_params['dataset']['corpus']
        ),
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
    log.info(f"Reading filtered word embeddings from: '{embeddings_path}' ...")
    if not os.path.isfile(embeddings_path):
        log.error(f"No such filtered word embeddings file: '{embeddings_path}'.")
    assert os.path.isfile(embeddings_path), 'Ensure word embeddings file exists.'
    word_embed, w2i = get_embeddings(
        embeddings_path,
        input_params['model']['embeddings_model'],
        input_params['model']['embeddings_size'],
        input_params['model']['embeddings_pretrained']
    )

    trainer = Train(w2i, word_embed, train, input_params)
    result = trainer.run_and_save()

    log.info(f"Training model total time = {((time.time() - initial_time) / 60)} minutes")
    log.info(f"MODULE EXECUTED.")
