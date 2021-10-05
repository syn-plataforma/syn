"""Tasks module"""
import os
import time
from pathlib import Path

import numpy as np
from sklearn import metrics
from tqdm import tqdm

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.helpers.treelstm.dataloader import DataLoader
from syn.model.build.common.task import Task, Experiment
from syn.model.build.treelstm.dynetconfig import get_dynet
from syn.model.build.treelstm.dynetmodel import get_dynet_model

load_environment_variables()
log = set_logger()

dy = get_dynet()


def check_data_integrity(trees: list = None, attention_vectors: list = None) -> None:
    num_trees = len(trees)
    num_attn_vectors = len(attention_vectors)
    assert num_trees == num_attn_vectors, \
        f"Distinct length of trees '{num_trees}' and attention vectors '{num_attn_vectors}'."


class Train(Task):
    def __init__(
            self,
            w2i: dict = None,
            word_embed: np.ndarray = None,
            train: DataLoader = None,
            dev: DataLoader = None,
            params: dict = None
    ):
        super().__init__()
        self.params = params.copy()
        self.kwargs = {
            'scheduler': self.params['scheduler'],
            'model': self.params['model'],
            'dataset': self.params['dataset']
        }
        self.task_id = self.get_hash()
        self.task_name = f"{self.params['dataset']['task']}-{self.params['dataset']['corpus']}-" \
                         f"{self.params['model']['architecture']}-{self.params['alias']}"
        self.save_dbname = 'tasks'
        self.save_collection = 'experiments'

        # Save params.
        # if not os.path.isfile(str(Path(self.params['scheduler']['model_save_dir']) / 'params.json')):
        self.save_dict_to_json(self.params, str(Path(self.params['scheduler']['model_save_dir']) / 'params.json'))

        # Model.
        self.model = get_dynet_model(self.params['dataset']['task'])(
            n_classes=self.params['model']['n_classes'],
            w2i=w2i,
            word_embed=word_embed,
            params=self.params['model'],
            model_meta_file=self.params['model_meta_file'] if 'model_meta_file' in self.params else ''
        )

        # Datasets.
        self.train, self.dev = train, dev

        self.trainer_param = getattr(dy, self.params['scheduler']['trainer'])(self.model.pc_param)
        self.trainer_embed = getattr(dy, self.params['scheduler']['trainer'])(self.model.pc_embed)

        self.trainer_param.learning_rate = self.params['scheduler']['learning_rate_param']
        self.trainer_embed.learning_rate = self.params['scheduler']['learning_rate_embed']

        all_trainers = [self.trainer_param, self.trainer_embed]
        for trainer in all_trainers:
            trainer.set_clip_threshold(-1)
            trainer.set_sparse_updates(self.params['scheduler']['sparse'])

    def load(self):
        raise NotImplementedError("This method must be inherited")

    def run(self) -> dict:
        log.info(f"Training model ...")
        initial_time = time.time()
        time_stamp = time.time()
        total_time = []
        best_acc = 0
        n_endure = 0
        endure_upper = int(os.environ.get('ENDURE_UPPER', 10))
        model_meta_file = None
        num_epochs = 0
        epochs_acc_curve_x = []
        epochs_acc_curve_y = []
        log.info(f"Maximum number of epochs to train model: {self.kwargs['scheduler']['max_epochs']}.")
        for i in range(int(self.kwargs['scheduler']['max_epochs'])):
            log.info(f"Epoch {i} ...")
            self.train.reset()
            loss = 0.0
            time_start = time.time()
            for j, batch in enumerate(self.train.batches(batch_size=self.params['scheduler']['batch_size']), 1):
                # build graph for this batch instance
                dy.renew_cg()

                # Logits
                loss_expr = self.model.losses_batch(batch, self.params['scheduler']['loss_function'])

                loss += loss_expr.value()  # this performs a forward through the network.

                # now do an optimization step
                loss_expr.backward()  # compute the gradients

                # loss = self.model.losses_batch(batch, self.params['scheduler']['loss_function'])
                # if self.params['scheduler']['regularization_strength'] > 0:
                #     loss += self.model.regularization_loss(coef=self.params['scheduler']['regularization_strength'])
                #
                # # this performs a forward through the network.
                # loss_value = loss.value()
                #
                # # now do an optimization step
                # loss.backward()  # compute the gradients

                self.trainer_param.update()
                self.trainer_embed.update()

                if j % 50 == 0:
                    self.trainer_param.status()
                    self.trainer_embed.status()
                    total = self.train.n_samples / self.params['scheduler']['batch_size']
                    last_batch = total if total % 50 == 0 else total - 50
                    print(f"j: {j}/{last_batch} - loss_value: {loss}")

            time_epoch = (time.time() - time_start) / 60
            total_time.append(time_epoch)
            log.info(f"Epoch {i} total time {time_epoch}")
            self.trainer_param.learning_rate *= self.params['scheduler']['learning_rate_decay']
            self.trainer_embed.learning_rate *= self.params['scheduler']['learning_rate_decay']

            acc = self.accuracy_evaluation() if self.params['dataset']['task'] not in ['duplicity', 'similarity'] \
                else self.duplicity_accuracy_evaluation()
            best_acc, updated = max(acc, best_acc), acc > best_acc
            log.info(f"dev_acc={acc} - best_dev_acc={best_acc}")
            epochs_acc_curve_x.append(i)
            epochs_acc_curve_y.append(acc)

            if updated:
                tic = time.time()
                model_name = str(time_stamp) + '_' + str(i)
                log.info(f"Deleting saved model in '{model_meta_file}' ...")
                self.model.delete(model_meta_file)
                model_meta_file = self.model.save(self.params['scheduler']['model_save_dir'], model_name)
                log.info(f"Updating saved model in '{model_meta_file}' ...")
                log.info(f"Updated saved model total time: {(time.time() - tic) / 60} minutes.")
                n_endure = 0
            else:
                n_endure += 1
                if n_endure > endure_upper:
                    break

            num_epochs = i

        # Return task.
        self.task_action = {
            'kwargs': self.kwargs,
            'train': {
                'type': self.__class__.__name__,
                'model_meta_file': model_meta_file,
                'best_acc': best_acc,
                'epochs': num_epochs,
                'epochs_acc_curve': {'x': epochs_acc_curve_x, 'y': epochs_acc_curve_y},
                'training_time_minutes': (time.time() - initial_time) / 60
            }
        }

        return self.task_action

    def accuracy_evaluation(self):
        self.dev.reset()
        good = bad = 0.0
        preds = set()
        labels = set()

        for inst in tqdm(self.dev, total=self.dev.n_samples, desc='rows'):
            # Tuple(inst[0], inst[1], inst[2]).
            # Tuple(trees, attention_vectors, label).

            # build graph for this instance
            dy.renew_cg()

            # Check data integrity.
            check_data_integrity(inst[0], inst[1])

            # Issue description as Tuple(trees, attention_vectors).
            issue_description = (inst[0], inst[1])

            # Issue structured data.
            issue_structured_data = inst[2]

            logits, _ = self.model.classify(issue_description, issue_structured_data)
            pred = np.argmax(logits.npvalue())
            preds.add(pred)
            labels.add(inst[3])
            if pred == inst[3]:
                good += 1
            else:
                bad += 1
        acc = good / (good + bad)
        log.info(f"Predictions: {preds} - Labels: {labels}")
        return acc

    def duplicity_accuracy_evaluation(self):
        self.dev.reset()
        good = bad = 0.0
        preds = set()
        labels = set()

        for inst in tqdm(self.dev, total=self.dev.n_samples, desc='rows'):
            # Tuple(inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6]) =
            # Tuple(trees_left, trees_right, attention_vectors_left, attention_vectors_right, structured_data_left,
            # structured_data_right, label).

            # build graph for this instance
            dy.renew_cg()

            # Check data integrity.
            check_data_integrity(inst[0], inst[2])
            check_data_integrity(inst[1], inst[3])

            # Issue description as Tuple(trees, attention_vectors).
            issue_description_left = (inst[0], inst[2])
            issue_description_right = (inst[1], inst[3])

            # Issue structured data.
            issue_structured_data_left = inst[4]
            issue_structured_data_right = inst[5]

            # Compute logits
            logits, _, _ = self.model.classify(
                [issue_description_left, issue_description_right],
                [issue_structured_data_left, issue_structured_data_right]
            )

            # Get prediction
            pred = np.argmax(logits.npvalue())
            preds.add(pred)
            labels.add(inst[6])
            if pred == inst[6]:
                good += 1
            else:
                bad += 1
        acc = good / (good + bad)
        log.info(f"Predictions: {preds} - Labels: {labels}")
        return acc


class Evaluate(Experiment):
    def __init__(self, test, w2i, word_embed, params):
        model_builder = get_dynet_model(params['dataset']['task'])
        super().__init__(model_builder, test, w2i, word_embed, params)

    def run(self) -> dict:
        initial_time = time.time()
        calculated_metrics = self.multiclass_metrics_evaluation() \
            if self.params['dataset']['task'] not in ['duplicity', 'similarity'] else self.binary_metrics_evaluation()

        self.task_action['evaluation'] = {
            'type': self.__class__.__name__,
            'metrics': calculated_metrics,
            'evaluation_time': (time.time() - initial_time) / 60
        }

        return self.task_action

    def binary_metrics_evaluation(self):
        self.test.reset()
        preds = set()
        labels = set()

        y_pred = []
        y_true = []
        y_predict_proba = []
        for inst in tqdm(self.test, total=self.test.n_samples, desc='rows'):
            # Tuple(inst[0], inst[1], inst[2], inst[3], inst[4]) =
            # Tuple(trees_left, trees_right, attention_vectors_left, attention_vectors_right, label).

            # build graph for this instance
            dy.renew_cg()

            # Check data integrity.
            check_data_integrity(inst[0], inst[2])
            check_data_integrity(inst[1], inst[3])

            # Issue description as Tuple(trees, attention_vectors).
            issue_description_left = (inst[0], inst[2])
            issue_description_right = (inst[1], inst[3])

            # Issue structured data.
            issue_structured_data_left = inst[4]
            issue_structured_data_right = inst[5]

            # Get prediction
            pred, predict_proba, _, _ = self.model.predict(
                issue_description_left, issue_description_right,
                issue_structured_data_left, issue_structured_data_right
            )

            y_pred.append(int(pred))
            y_predict_proba.append(predict_proba)
            label = inst[6]
            y_true.append(int(label))

            # Distinct predictions and labels.
            preds.add(pred)
            labels.add(inst[6])

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

        good = bad = 0.0
        preds = set()
        labels = set()

        y_pred = []
        y_true = []
        y_predict_proba = []
        for inst in tqdm(self.test, total=self.test.n_samples, desc='rows'):
            # Tuple(inst[0], inst[1], inst[2], inst[3]).
            # Tuple(trees, attention_vectors, structured_data, label).

            # build graph for this instance
            dy.renew_cg()

            # Check data integrity.
            check_data_integrity(inst[0], inst[1])

            # Issue description as Tuple(trees, attention_vectors).
            issue_description = (inst[0], inst[1])

            # Issue structured data.
            issue_structured_data = inst[2]

            pred, predict_proba, _ = self.model.predict(issue_description, issue_structured_data)
            y_pred.append(int(pred))
            y_predict_proba.append(predict_proba)
            label = inst[3]
            y_true.append(int(label))

            # Distinct predictions and labels.
            preds.add(pred)
            labels.add(inst[3])

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
