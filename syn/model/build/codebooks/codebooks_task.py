"""Tasks module"""
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

from syn.helpers.environment import load_environment_variables
from syn.helpers.hyperparams import get_codebooks_kwargs_from_params
from syn.helpers.logging import set_logger
from syn.helpers.treelstm.dataloader import DataLoader
from syn.model.build.codebooks.codebooks_model import get_codebooks_model, CodebooksBuilder, DataSplitter, \
    WordEmbeddingsTransformer
from syn.model.build.common.task import Task, Experiment, get_task_kwargs

load_environment_variables()
log = set_logger()


class ClassifierCodebooksTrain(Task):
    def __init__(
            self,
            w2i: dict = None,
            word_embed: np.ndarray = None,
            train: DataLoader = None,
            params: dict = None
    ):
        super().__init__()
        self.w2i = w2i
        self.word_embed = word_embed
        self.train = train
        self.params = params.copy()

        self.kwargs = self.params
        self.task_id = self.get_hash()
        self.save_dbname = 'tasks'
        self.save_collection = 'codebooks'
        self.task = self._db_load()

        task_name = f"{self.params['parent_class']}-{self.params['dataset']['task']}-" \
                    f"{self.params['dataset']['corpus']}-{self.params['embeddings']['model']}-" \
                    f"{self.params['embeddings']['size']}-" \
                    f"{'pretrained' if self.params['embeddings']['pretrained'] else 'trained'}-" \
                    f"{self.params['codebooks']['n_codewords']}"
        self.task_name = self.task['task_name'] if self.task is not None and 'task_name' in self.task else task_name
        self.task_action = self.task['task_action'] if self.task is not None and 'task_action' in self.task else None

        # Model.
        self.model = CodebooksBuilder(params=self.params['codebooks'])

        self.model_meta_file = self._get_model_meta_file()
        if self.model_meta_file is not None and self.model_meta_file != '':
            self.load()

    def _get_model_meta_file(self):
        model_meta_file = None
        if self.task_action is not None and 'train' in self.task_action and 'model_meta_file' \
                in self.task_action['train']:
            model_meta_file = self.task_action['train']['model_meta_file']

        return model_meta_file

    def save(self, save_dir, model_name):
        model_path = os.path.join(save_dir, model_name + '.pkl')

        log.info(f"Saving model to: '{model_path}'.")
        tic = time.time()
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file, pickle.HIGHEST_PROTOCOL)
        log.info(f"Saving model total time: {(time.time() - tic) / 60} minutes")

        return model_path

    def load(self):
        log.info(f"Loading model from: '{self.model_meta_file}'.")
        tic = time.time()
        with open(self.model_meta_file, 'rb') as model_file:
            saved_model: CodebooksBuilder = pickle.load(model_file)

        self.model = saved_model
        log.info(f"Loading model total time: {(time.time() - tic) / 60} minutes")

    def run(self) -> dict:
        if self.model.codebooks is None:
            log.info(f"Training model ...")

            # reset data
            self.train.reset()

            # split train data
            issue_description, _, _ = DataSplitter().transform(
                self.train, self.params['dataset']['task']
            )

            # clear variables to free used memory
            self.train = None

            # fit codebooks model using vocabulary word embeddings
            self.model.fit(self.word_embed)

            # Word embeddings transformer
            word_embeddings_transformer = WordEmbeddingsTransformer()
            log.info(f"Transforming issues descriptions to word embeddings ...")
            tic = time.time()
            word_embeddings = word_embeddings_transformer.transform(issue_description, self.w2i, self.word_embed)
            log.info(
                f"Transforming issues descriptions to word embeddings total time = {(time.time() - tic) / 60} minutes")

            # clear variables to free used memory
            issue_description = None
            self.word_embed = None
            self.w2i = None

            # transform word embeddings to codebooks
            log.info(f"Transforming word embeddings to codebooks ...")
            tic = time.time()
            self.model.codebooks = self.model.transform(word_embeddings, as_string=True)
            log.info(f"Transforming word embeddings to codebooks total time = {(time.time() - tic) / 60} minutes")

            # clear variables to free used memory
            word_embeddings = None

        # save codebooks model
        model_path = Path(self.params['scheduler']['model_save_dir']) / 'codebooks'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_meta_file = self.save(model_path, self.task_name)

        # Return codebooks info
        self.task_action = {
            'kwargs': self.kwargs,
            'train': {
                'type': self.__class__.__name__,
                'model_meta_file': model_meta_file
            }
        }

        return self.task_action


class SimilarityMeterCodebooksTrain(Task):
    def __init__(
            self,
            w2i: dict = None,
            word_embed: np.ndarray = None,
            train: DataLoader = None,
            params: dict = None
    ):
        super().__init__()
        self.w2i = w2i
        self.word_embed = word_embed
        self.train = train
        self.params = params.copy()

        self.kwargs = self.params
        model_path = Path(self.kwargs['scheduler']['model_save_dir']) / 'codebooks'
        self.kwargs['scheduler']['model_save_dir'] = str(model_path)
        self.task_id = self.get_hash()

        self.save_dbname = 'tasks'
        self.save_collection = 'codebooks'
        self.task = self._db_load()

        task_name = f"{self.params['parent_class']}-{self.params['dataset']['task']}-" \
                    f"{self.params['dataset']['corpus']}-{self.params['embeddings']['model']}-" \
                    f"{self.params['embeddings']['size']}-" \
                    f"{'pretrained' if self.params['embeddings']['pretrained'] else 'trained'}-" \
                    f"{self.params['codebooks']['n_codewords']}"
        self.task_name = self.task['task_name'] if self.task is not None and 'task_name' in self.task else task_name
        self.task_action = self.task['task_action'] if self.task is not None and 'task_action' in self.task else None

        # Model.
        self.model = CodebooksBuilder(params=self.params['codebooks'])

        self.model_meta_file = self._get_model_meta_file()
        if self.model_meta_file is not None and self.model_meta_file != '':
            self.load()

    def _get_model_meta_file(self):
        model_meta_file = None
        if self.task_action is not None and 'train' in self.task_action and 'model_meta_file' \
                in self.task_action['train']:
            model_meta_file = self.task_action['train']['model_meta_file']

        return model_meta_file

    def save(self, save_dir, model_name):
        model_path = os.path.join(save_dir, model_name + '.pkl')

        log.info(f"Saving model to: '{model_path}'.")
        tic = time.time()
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file, pickle.HIGHEST_PROTOCOL)
        log.info(f"Saving model total time: {(time.time() - tic) / 60} minutes")

        return model_path

    def load(self):
        log.info(f"Loading model from: '{self.model_meta_file}'.")
        tic = time.time()
        with open(self.model_meta_file, 'rb') as model_file:
            saved_model: CodebooksBuilder = pickle.load(model_file)

        self.model = saved_model
        log.info(f"Loading model total time: {(time.time() - tic) / 60} minutes")

    def run(self) -> dict:
        if self.model.codebooks is None:
            log.info(f"Training model ...")

            # reset data
            self.train.reset()

            # split train data
            issue_description_left, issue_description_right, _, _, _ = DataSplitter().transform(
                self.train,
                self.params['dataset']['task']
            )

            # clear variables to free used memory
            self.train = None

            # fit codebooks model using vocabulary word embeddings
            self.model.fit(self.word_embed)

            # use left and right issues descriptions
            log.info(f"Concatenating left and right issues descriptions ...")
            tic = time.time()
            issue_description_total = list(np.concatenate(
                (
                    np.array(issue_description_left, dtype="object"),
                    np.array(issue_description_right, dtype="object"))
                , axis=0)
            )
            log.info(f"Concatenating left and right issues descriptions "
                     f"total time = {(time.time() - tic) / 60} minutes")

            # clear variables to free used memory
            issue_description_left = None
            issue_description_right = None

            # Word embeddings transformer
            word_embeddings_transformer = WordEmbeddingsTransformer()
            log.info(f"Transforming issues descriptions to word embeddings ...")
            tic = time.time()
            word_embeddings = word_embeddings_transformer.transform(issue_description_total, self.w2i, self.word_embed)
            log.info(
                f"Transforming issues descriptions to word embeddings total time = {(time.time() - tic) / 60} minutes")

            # clear variables to free used memory
            issue_description_total = None
            self.word_embed = None
            self.w2i = None

            # transform word embeddings to codebooks
            log.info(f"Transforming word embeddings to codebooks ...")
            tic = time.time()
            self.model.codebooks = self.model.transform(word_embeddings, as_string=True)
            log.info(f"Transforming word embeddings to codebooks total time = {(time.time() - tic) / 60} minutes")

            # clear variables to free used memory
            word_embeddings = None

        # save codebooks model
        model_path = Path(self.params['scheduler']['model_save_dir']) / 'codebooks'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_meta_file = self.save(model_path, self.task_name)

        # Return codebooks info
        self.task_action = {
            'kwargs': self.kwargs,
            'train': {
                'type': self.__class__.__name__,
                'model_meta_file': model_meta_file
            }
        }

        return self.task_action


class Train(Task):
    def __init__(
            self,
            w2i: dict = None,
            word_embed: np.ndarray = None,
            train: DataLoader = None,
            params: dict = None
    ):
        super().__init__()
        self.w2i = w2i
        self.word_embed = word_embed
        self.train = train
        self.params = params.copy()
        self.kwargs = get_task_kwargs(self.params)
        self.task_id = self.get_hash()

        self.task_name = f"{self.params['dataset']['task']}-{self.params['dataset']['corpus']}-" \
                         f"{self.params['model']['architecture']}-{self.params['alias']}"
        self.save_dbname = 'tasks'
        self.save_collection = 'experiments'

        # Save params.
        self.save_dict_to_json(self.params, str(Path(self.params['scheduler']['model_save_dir']) / 'params.json'))

        # codebooks trainer
        codebooks_kwargs = self.params.copy()
        codebooks_kwargs['parent_class'] = self.__class__.__name__.lower()
        if self.params['dataset']['task'] in ['duplicity', 'similarity']:
            self.codebooks_trainer = SimilarityMeterCodebooksTrain(
                params=get_codebooks_kwargs_from_params(codebooks_kwargs)
            )
        else:
            self.codebooks_trainer = ClassifierCodebooksTrain(params=get_codebooks_kwargs_from_params(codebooks_kwargs))

        self.codebooks_trainer = ClassifierCodebooksTrain(params=get_codebooks_kwargs_from_params(codebooks_kwargs))
        if self.codebooks_trainer.model.codebooks is not None:
            log.info(f"Loaded codebooks from '{self.codebooks_trainer.model_meta_file}'")
            self.params['model']['codebooks_model_meta_file'] = self.codebooks_trainer.model_meta_file

        # Model.
        self.model = get_codebooks_model(self.params['dataset']['task'])(
            params=self.params,
            model_meta_file=self.params['model_meta_file'] if 'model_meta_file' in self.params else None
        )

    def load(self):
        raise NotImplementedError("This method must be inherited")

    def run(self) -> dict:
        log.info(f"Training model ...")
        initial_time = time.time()

        # reset data
        self.train.reset()

        # model fitting
        self.model.fit(self.train, self.w2i, self.word_embed)

        # clear variables to free used memory
        self.train = None

        # save codebooks
        codebooks_model_meta_file = self.codebooks_trainer.model_meta_file
        if self.codebooks_trainer.model.codebooks is None:
            self.codebooks_trainer.model = self.model.issue_description_builder.codebooks_builder
            result = self.codebooks_trainer.run_and_save()
            codebooks_model_meta_file = result['train']['model_meta_file']

        # clear variables to free used memory
        self.codebooks_trainer = None
        self.model.issue_description_builder.codebooks_builder.codebooks = None

        # save classifier model
        model_meta_file = None
        model_path = Path(self.params['scheduler']['model_save_dir'])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for subdir in ['meta', 'model']:
            if not os.path.exists(Path(model_path) / subdir):
                os.makedirs(Path(model_path) / subdir)

        self.model.delete(model_meta_file)
        model_meta_file = self.model.save(model_path, 'params', 'model')

        # remove codebooks_model_meta_file from kwargs
        if 'codebooks_model_meta_file' in self.kwargs['model'].keys():
            del self.kwargs['model']['codebooks_model_meta_file']

        # Return task.
        self.task_action = {
            'kwargs': self.kwargs,
            'train': {
                'type': self.__class__.__name__,
                'model_meta_file': model_meta_file,
                'codebooks_model_meta_file': codebooks_model_meta_file,
                'training_time_minutes': (time.time() - initial_time) / 60
            }
        }

        return self.task_action


class Evaluate(Experiment):
    def __init__(self, test, w2i, word_embed, params):
        model_builder = get_codebooks_model(params['dataset']['task'])
        super().__init__(model_builder, test, w2i, word_embed, params)

        # codebooks trainer
        codebooks_kwargs = self.params.copy()
        codebooks_kwargs['parent_class'] = self.__class__.__name__.lower()
        self.codebooks_trainer = ClassifierCodebooksTrain(params=get_codebooks_kwargs_from_params(codebooks_kwargs))
        if self.codebooks_trainer.model.codebooks is not None:
            self.model.issue_description_builder.codebooks_builder = self.codebooks_trainer.model
            self.params['model']['codebooks_model_meta_file'] = self.codebooks_trainer.model_meta_file

    def run(self) -> dict:
        initial_time = time.time()

        calculated_metrics = self.multiclass_metrics_evaluation() \
            if self.params['dataset']['task'] not in ['duplicity', 'similarity'] else self.binary_metrics_evaluation()

        # save codebooks
        codebooks_model_meta_file = self.codebooks_trainer.model_meta_file
        if self.codebooks_trainer.model.codebooks is None:
            self.codebooks_trainer.model = self.model.issue_description_builder.codebooks_builder
            result = self.codebooks_trainer.run_and_save()
            codebooks_model_meta_file = result['train']['model_meta_file']

        # remove codebooks_model_meta_file from kwargs
        if 'codebooks_model_meta_file' in self.kwargs['model'].keys():
            del self.kwargs['model']['codebooks_model_meta_file']

        self.task_action['evaluation'] = {
            'type': self.__class__.__name__,
            'codebooks_model_meta_file': codebooks_model_meta_file,
            'metrics': calculated_metrics,
            'evaluation_time': (time.time() - initial_time) / 60
        }

        return self.task_action

    def binary_metrics_evaluation(self):
        self.test.reset()

        y_pred, y_predict_proba, y_true = self.model.predict(self.test, self.w2i, self.word_embed)
        preds = set(y_pred)
        labels = set(y_true)

        log.info(f"Predictions: {preds} - Labels: {labels}")
        return {
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
            'average_precision': metrics.average_precision_score(y_true, y_pred),
            'neg_brier_score': metrics.brier_score_loss(y_true, y_pred),
            'f1': metrics.f1_score(y_true, y_pred, average='binary'),
            'neg_log_loss': metrics.log_loss(y_true, y_predict_proba),
            'precision': metrics.precision_score(y_true, y_pred, average='binary'),
            'recall': metrics.recall_score(y_true, y_pred, average='binary'),
            'jaccard': metrics.jaccard_score(y_true, y_pred, average='binary'),
            'roc_auc': metrics.roc_auc_score(y_true, y_pred),
            'confusion_matrix': metrics.confusion_matrix(y_true, y_pred).tolist()
        }

    def multiclass_metrics_evaluation(self):
        self.test.reset()

        y_pred, y_predict_proba, y_true = self.model.predict(self.test, self.w2i, self.word_embed)
        preds = set(y_pred)
        labels = set(y_true)

        log.info(f"Predictions: {preds} - Labels: {labels}")
        # Number of classes in y_true not equal to the number of columns in 'y_score'
        try:
            roc_auc_score = metrics.roc_auc_score(y_true, y_predict_proba, multi_class='ovr')
        except ValueError:
            log.error(f"Error calculating roc_auc_score.")
            roc_auc_score = float('NaN')
        try:
            roc_auc_ovo = metrics.roc_auc_score(y_true, y_predict_proba, multi_class='ovo')
        except ValueError:
            log.error(f"Error calculating roc_auc_score.")
            roc_auc_ovo = float('NaN')
        try:
            roc_auc_ovr_weighted = metrics.roc_auc_score(y_true, y_predict_proba, average='weighted', multi_class='ovr')
        except ValueError:
            log.error(f"Error calculating roc_auc_score.")
            roc_auc_ovr_weighted = float('NaN')
        try:
            roc_auc_ovo_weighted = metrics.roc_auc_score(y_true, y_predict_proba, average='weighted', multi_class='ovo')
        except ValueError:
            log.error(f"Error calculating roc_auc_score.")
            roc_auc_ovo_weighted = float('NaN')

        return {
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
            'precision_micro': metrics.precision_score(y_true, y_pred, average='micro'),
            'precision_macro': metrics.precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': metrics.precision_score(y_true, y_pred, average='weighted'),
            'recall_micro': metrics.recall_score(y_true, y_pred, average='micro'),
            'recall_macro': metrics.recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': metrics.recall_score(y_true, y_pred, average='weighted'),
            'jaccard_micro': metrics.jaccard_score(y_true, y_pred, average='micro'),
            'jaccard_macro': metrics.jaccard_score(y_true, y_pred, average='macro'),
            'jaccard_weighted': metrics.jaccard_score(y_true, y_pred, average='weighted'),
            'f1_micro': metrics.f1_score(y_true, y_pred, average='micro'),
            'f1_macro': metrics.f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': metrics.f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': metrics.confusion_matrix(y_true, y_pred).tolist(),
            'hamming_loss': metrics.hamming_loss(y_true, y_pred),
            'roc_auc_ovr': roc_auc_score,
            'roc_auc_ovo': roc_auc_ovo,
            'roc_auc_ovr_weighted': roc_auc_ovr_weighted,
            'roc_auc_ovo_weighted': roc_auc_ovo_weighted
        }


class IssueAssignerTrain(Task):
    def __init__(
            self,
            train: pd.DataFrame = None,
            opened: pd.DataFrame = None,
            params: dict = None
    ):
        super().__init__()
        self.train = train
        self.opened = opened
        self.params = params.copy()
        self.kwargs = get_task_kwargs(self.params)
        self.task_id = self.get_hash()

        self.task_name = f"{self.params['dataset']['task']}-{self.params['dataset']['corpus']}-" \
                         f"{self.params['model']['architecture']}-{self.params['alias']}"
        self.save_dbname = 'tasks'
        self.save_collection = 'experiments'

        # Save params.
        self.save_dict_to_json(self.params, str(Path(self.params['scheduler']['model_save_dir']) / 'params.json'))

        # Model.
        self.model = get_codebooks_model(self.params['dataset']['task'])(
            params=self.params,
            model_meta_file=self.params['model_meta_file'] if 'model_meta_file' in self.params else None
        )

    def load(self):
        raise NotImplementedError("This method must be inherited")

    def run(self) -> dict:
        log.info(f"Training model ...")
        initial_time = time.time()

        # model fitting
        self.model.fit(self.train, self.opened)

        # clear variables to free used memory
        self.train = None
        self.opened = None

        # save assigner model
        model_meta_file = None
        model_path = Path(self.params['scheduler']['model_save_dir'])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for subdir in ['meta', 'model']:
            if not os.path.exists(Path(model_path) / subdir):
                os.makedirs(Path(model_path) / subdir)

        self.model.delete(model_meta_file)
        model_meta_file = self.model.save(model_path, 'params', 'model')

        # Return task.
        self.task_action = {
            'kwargs': self.kwargs,
            'train': {
                'type': self.__class__.__name__,
                'model_meta_file': model_meta_file,
                'training_time_minutes': (time.time() - initial_time) / 60
            }
        }

        return self.task_action


class IssueAssignerEvaluate(Experiment):
    def __init__(self, test, w2i, word_embed, params):
        model_builder = get_codebooks_model(params['dataset']['task'])
        super().__init__(model_builder, test, w2i, word_embed, params)

    def run(self) -> dict:
        initial_time = time.time()

        calculated_metrics = self.multiclass_metrics_evaluation()

        self.task_action['evaluation'] = {
            'type': self.__class__.__name__,
            'metrics': calculated_metrics,
            'evaluation_time': (time.time() - initial_time) / 60
        }

        return self.task_action

    @staticmethod
    def _get_valid_y_pred(devs: list = None, y: list = None, val: str = None) -> str:
        result = None
        for dev in devs:
            if dev in y:
                result = dev
                break
            else:
                y_set = set(y)
                y_set.remove(val)
                result = list(y_set)[0]
        return result

    def multiclass_metrics_evaluation(self):
        y_prediction, y_true = self.model.predict(self.test)

        # process predictions
        y_pred = []
        for y in zip(y_true, y_prediction):
            developers = [item['label'] for item in y[1]]
            if y[0] in developers:
                y_pred.append(y[0])
            else:
                y_pred.append(self._get_valid_y_pred(developers, y_true, y[0]))

        # labels and predictions
        predictions = set(y_pred)
        labels = set(y_true)
        log.info(f"Predictions: {predictions} - Labels: {labels}")

        print(predictions - labels)

        # metrics
        return {
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
            'precision_micro': metrics.precision_score(y_true, y_pred, average='micro'),
            'precision_macro': metrics.precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': metrics.precision_score(y_true, y_pred, average='weighted'),
            'recall_micro': metrics.recall_score(y_true, y_pred, average='micro'),
            'recall_macro': metrics.recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': metrics.recall_score(y_true, y_pred, average='weighted'),
            'jaccard_micro': metrics.jaccard_score(y_true, y_pred, average='micro'),
            'jaccard_macro': metrics.jaccard_score(y_true, y_pred, average='macro'),
            'jaccard_weighted': metrics.jaccard_score(y_true, y_pred, average='weighted'),
            'f1_micro': metrics.f1_score(y_true, y_pred, average='micro'),
            'f1_macro': metrics.f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': metrics.f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': metrics.confusion_matrix(y_true, y_pred).tolist(),
            'hamming_loss': metrics.hamming_loss(y_true, y_pred)
        }
