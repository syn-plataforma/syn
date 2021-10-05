import gc
import os
import pickle
import time
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union, Type

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from syn.helpers.assignation import get_developer_issues
from syn.helpers.assignation import jaccard_similarity
from syn.helpers.codebooks.dataloader import DataLoader
from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger

load_environment_variables()
log = set_logger()


class CodebooksModel(object):
    def __init__(self, params, model_meta_file=None):
        log.info(f"CodebooksModel init ...")
        self.params = params.copy() if params is not None else None

        # Load input params.
        if model_meta_file is not None and model_meta_file != '' and params is not None:
            log.info(f"Loading hyperparameters from: '{str(Path(model_meta_file))}'.")
            saved_params = np.load(str(Path(model_meta_file)), allow_pickle=True).item()
            if saved_params is not None:
                self.params.update(saved_params)
        log.info(f"CodebooksModel init end")

    def save(self, save_dir, params_name, model_name):
        meta_path = os.path.join(save_dir, 'meta', params_name)
        model_path = os.path.join(save_dir, 'model', model_name + '.pkl')

        log.info(f"Saving params to: '{meta_path}'.")
        np.save(meta_path, self.params)

        log.info(f"Saving model to: '{model_path}'.")
        tic = time.time()
        with open(model_path, 'wb') as model_file:
            pickle.dump(self, model_file, pickle.HIGHEST_PROTOCOL)
        log.info(f"Saving model total time: {(time.time() - tic) / 60} minutes")

        return meta_path + '.npy'

    def load_params_model(
            self, model_meta_file: str = None,
            destination_attribute: str = None,
            saved_attribute: str = None
    ):
        model_path = model_meta_file.replace('meta', 'model').replace('params.npy', 'model.pkl')
        tic = time.time()
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            log.info(f"Loading model from: '{model_path}'.")
            with open(model_path, 'rb') as model_file:
                saved_model: Union[IssueClassifier, IssueSimilarityMeter, IssueAssigner] = pickle.load(model_file)

            if destination_attribute is not None:
                if saved_attribute is not None:
                    if hasattr(saved_model, saved_attribute):
                        self.__setattr__(destination_attribute, saved_model.__getattribute__(saved_attribute))
                else:
                    self.__setattr__(destination_attribute, saved_model)

            log.info(f"Loading model total time: {(time.time() - tic) / 60} minutes")

    @staticmethod
    def delete(model_meta_file):
        if model_meta_file is None:
            return
        log.info(f"Deleting saved model in '{model_meta_file}' ...")
        os.remove(model_meta_file)
        model_meta_file = model_meta_file.replace('.npy', '')
        os.remove(model_meta_file.replace('meta', 'model'))


class Classifier(object):
    def __init__(self, params, model_meta_file=None):
        self.params = params.copy()

        # Load input params.
        if model_meta_file is not None and model_meta_file != '':
            log.info(f"Loading hyperparameters from: '{str(Path(model_meta_file))}'.")
            saved_params = np.load(str(Path(model_meta_file)), allow_pickle=True).item()
            self.params.update(saved_params)

        # Add parameters for issue description representation.
        self.model = None

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError("This method must be inherited")

    def save(self, save_dir, model_name):
        meta_path = os.path.join(save_dir, 'meta', model_name)
        model_path = os.path.join(save_dir, 'model', model_name + '.joblib')

        log.info(f"Saving params to: '{meta_path}'.")
        np.save(meta_path, self.params)

        log.info(f"Saving model to: '{model_path}'.")
        tic = time.time()
        dump(self.model, model_path)
        log.info(f"Saving model total time: {(time.time() - tic) / 60} minutes")

        return meta_path + '.npy'

    def load_param_embed(self, model_meta_file):
        model_meta_file = model_meta_file.replace('.npy', '')
        model_path = model_meta_file.replace('meta', 'model') + '.joblib'

        log.info(f"Loading model from: '{model_path}'.")
        tic = time.time()
        self.model = load(model_path)
        log.info(f"Loading model total time: {(time.time() - tic) / 60} minutes")

    @staticmethod
    def delete(model_meta_file):
        if model_meta_file is None:
            return
        os.remove(model_meta_file)
        model_meta_file = model_meta_file.replace('.npy', '')
        os.remove(model_meta_file.replace('meta', 'model'))


class DataSplitter(object):
    @staticmethod
    def fit(data: DataLoader = None, task: str = None):
        return DataSplitter.transform(data, task)

    @staticmethod
    def _transform_data_for_classifier(data):
        tic = time.time()
        issue_description = []
        issue_structured_data = []
        label = []

        result = None
        for inst in tqdm(data.data, total=data.n_samples, desc='rows'):
            # Tuple(inst[0], inst[1], inst[2]) = Tuple(tokens, structured_data, label).

            # Issue description.
            issue_description.append(inst[0][0])

            # Issue structured data
            issue_structured_data.append([elem[0] for elem in inst[1]])

            # label
            label.append(inst[2])

            result = issue_description, issue_structured_data, label

        log.info(f"Splitting data total time = {(time.time() - tic) / 60} minutes")

        return issue_description, issue_structured_data, label

    @staticmethod
    def _transform_data_for_similarity(data):
        tic = time.time()
        issue_description_left = []
        issue_structured_data_left = []
        issue_description_right = []
        issue_structured_data_right = []
        label = []

        for inst in tqdm(data.data, total=data.n_samples, desc='rows'):
            # Tuple(inst[0], inst[1], inst[2], inst[3], inst[4]) =
            # Tuple(tokens_left, tokens_right, structured_data_left, structured_data_right, label).

            # Issues descriptions.
            issue_description_left.append(inst[0][0])
            issue_description_right.append(inst[1][0])

            # Issues structured data
            issue_structured_data_left.append([elem[0] for elem in inst[2]])
            issue_structured_data_right.append([elem[0] for elem in inst[3]])

            # label
            label.append(inst[4])

        log.info(f"Splitting data total time = {(time.time() - tic) / 60} minutes")

        return issue_description_left, issue_description_right, issue_structured_data_left, issue_structured_data_right, label

    @staticmethod
    def transform(data: DataLoader = None, task: str = None):
        log.info(f"Splitting data for task '{task}' ...")
        tic = time.time()
        result = None

        if task in ['duplicity', 'similarity']:
            result = DataSplitter._transform_data_for_similarity(data)
        else:
            result = DataSplitter._transform_data_for_classifier(data)

        log.info(f"Splitting data total time = {(time.time() - tic) / 60} minutes")

        return result


class WordEmbeddingsTransformer(object):
    def fit(self, data, w2i, word_embed):
        return self.transform(data, w2i, word_embed)

    @staticmethod
    def transform(data, w2i, word_embed):
        result = []
        for sentence in tqdm(data, total=len(data), desc='rows'):
            sentence_emb = []
            for word in sentence:
                emb = word_embed[w2i.get(word, 0)]
                sentence_emb.append(emb)
            result.append(sentence_emb)

        return np.array(result, dtype="object")


class CodebooksBuilder(CodebooksModel):
    def __init__(
            self,
            params: dict = None,
            model_meta_file: str = None
    ):
        log.info(f"CodebooksBuilder init ...")
        super().__init__(params, model_meta_file)
        self.params = params.copy()
        self.n_codewords = self.params['n_codewords']

        # Make sure to always use a fixed seed so that results are reproducibles.
        self.model = cluster.MiniBatchKMeans(
            n_clusters=self.n_codewords,
            random_state=230,
            compute_labels=False
        )  # Labels are not needed

        self.codebooks = None

        if model_meta_file is not None and model_meta_file != '':
            log.info(f"Loading codebooks model from '{model_meta_file}'")
            self.load_params_model(model_meta_file, 'model', 'model')
        log.info(f"CodebooksBuilder init end")

    def fit(self, word_embed):
        log.info(f"Training codebooks model ...")
        # Stores the execution start time to calculate the time it takes for the module to execute.
        tic = time.time()

        # Model fitting.
        self.model.fit(word_embed)

        log.info(f"Training codebooks model total time = {(time.time() - tic) / 60} minutes")

    def inverse_transform(self, data):
        raise NotImplementedError

    def predict(self, data, as_string: bool = False):
        result = []
        num_batches = len(data)

        for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
            codebook = self.model.predict(np.array(data[idx]))
            codebook_str = ''
            if as_string:
                for cb_str in codebook:
                    codebook_str += str(cb_str) + ' '
            result.append(codebook if not as_string else codebook_str)

        return result

    def transform(self, data, as_string: bool = False):
        return self.predict(data, as_string)


class IssueDescriptionBuilder(CodebooksModel):
    def __init__(
            self,
            params: dict = None,
            model_meta_file: str = None
    ):
        log.info(f"IssueDescriptionBuilder init ...")
        super().__init__(params, model_meta_file)
        self.params = params.copy()

        # Word embeddings transformer
        self.word_embeddings_transformer = WordEmbeddingsTransformer()

        # Codebooks builder.
        self.codebooks_builder = None
        codebooks_params = self._build_params_from_dict('codebooks')
        self.codebooks_builder = CodebooksBuilder(params=codebooks_params)
        if 'model_meta_file' in codebooks_params.keys():
            self.load_params_model(codebooks_params.pop('model_meta_file'), 'codebooks_builder', None)
        # Issue description builder.
        tfidf_kwargs = self._build_params_from_dict('tfidf')

        self.model = TfidfVectorizer(lowercase=False, **tfidf_kwargs)
        if model_meta_file is not None and model_meta_file != '':
            self.load_params_model(model_meta_file, 'model', 'model')
        log.info(f"IssueDescriptionBuilder init end")

    def _build_params_from_dict(self, prefix):
        result = {}
        for param_name in self.params.keys():
            if str(param_name).startswith(f"{prefix}_"):
                result[param_name.replace(f"{prefix}_", '')] = self.params[param_name]

        return result

    def fit(self, data, w2i, word_embed):
        log.info(f"Training issue description builder ...")
        # Stores the execution start time to calculate the time it takes for the module to execute.

        if self.codebooks_builder.codebooks is None:
            # word embeddings
            log.info(f"Transforming issues descriptions to word embeddings ...")
            tic = time.time()
            word_embeddings = self.word_embeddings_transformer.transform(data, w2i, word_embed)
            log.info(f"Transforming issues descriptions to word embeddings "
                     f"total time = {(time.time() - tic) / 60} minutes")

            # fit codebooks model using vocabulary word embeddings
            self.codebooks_builder.fit(word_embed)

            # transform word embeddings to codebooks
            log.info(f"Transforming word embeddings to codebooks ...")
            tic = time.time()
            self.codebooks_builder.codebooks = self.codebooks_builder.transform(word_embeddings, as_string=True)
            log.info(f"Transforming word embeddings to codebooks total time = {(time.time() - tic) / 60} minutes")
        else:
            log.info(f"Using saved codebooks.")

        # TF-IDF features
        log.info(f"Training TF-IDF ...")
        tic = time.time()
        self.model.fit(self.codebooks_builder.codebooks)
        log.info(f"Training TF-IDF total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Training issue description builder total time = {(time.time() - tic) / 60} minutes")

    def fit_transform(self, data, w2i, word_embed):
        self.fit(data, w2i, word_embed)

        # TF-IDF features
        log.info(f"Transforming codebooks to document-term matrix ...")
        tic = time.time()
        # scipy.sparse.csr.csr_matrix
        result = self.model.transform(self.codebooks_builder.codebooks)
        log.info(f"Transforming codebooks to document-term matrix total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming issue description total time = {(time.time() - tic) / 60} minutes")

        return result

    def transform(self, data, w2i, word_embed):
        log.info(f"Transforming issue description ...")
        if self.codebooks_builder.codebooks is None:
            # word embeddings
            tic = time.time()
            log.info(f"Transforming issues descriptions to word embeddings ...")
            word_embeddings = self.word_embeddings_transformer.transform(data, w2i, word_embed)
            log.info(
                f"Transforming issues descriptions to word embeddings total time = {(time.time() - tic) / 60} minutes")

            # codebooks
            log.info(f"Transforming word embeddings to codebooks ...")
            tic = time.time()
            self.codebooks_builder.codebooks = self.codebooks_builder.transform(word_embeddings, as_string=True)
            log.info(f"Transforming word embeddings to codebooks total time = {(time.time() - tic) / 60} minutes")
        else:
            log.info(f"Using saved codebooks.")

        # TF-IDF features
        log.info(f"Transform codebooks to document-term matrix ...")
        tic = time.time()
        result = self.model.transform(self.codebooks_builder.codebooks)
        log.info(f"Transform codebooks to document-term matrix total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming issue description total time = {(time.time() - tic) / 60} minutes")

        return result


class IssueClassifier(CodebooksModel):
    def __init__(
            self,
            params: dict = None,
            model_meta_file: str = None
    ):
        log.info(f"IssueClassifier init ...")
        super().__init__(params, model_meta_file)
        self.params = params.copy()

        # Issue description builder.
        self.issue_description_builder = IssueDescriptionBuilder(self.params['model'])

        # Classifier.
        classifier_name = self.params['model']['classifier']
        classifier_kwargs = get_codebooks_classifier_kwargs(classifier_name, self.params['model'])

        self.model = get_codebooks_classifier(classifier_name)(**classifier_kwargs)

        if model_meta_file is not None and model_meta_file != '':
            self.load_params_model(model_meta_file, 'model', 'model')
            self.load_params_model(model_meta_file, 'issue_description_builder', 'issue_description_builder')
        log.info(f"IssueClassifier init end")

    def fit(self, data, w2i, word_embed):
        log.info(f"Training IssueClassifier ...")

        # split train data
        issue_description, issue_structured_data, label = DataSplitter().transform(data, self.params['dataset']['task'])

        # clear variables to free used memory
        data = None

        # TF-IDF features
        log.info(f"Training issue description model and transforming issue description to document-term matrix ...")
        tic = time.time()
        issue_description_repr = self.issue_description_builder.fit_transform(
            issue_description, w2i, word_embed
        )
        log.info(f"Manual garbage collection: {gc.collect()}")
        log.info(f"Training issue description model and transforming issue description to document-term matrix "
                 f"total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming issue description to document-term matrix "
                 f"total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array ...")
        tic = time.time()

        issue_description_repr_dense = []
        num_rows = issue_description_repr.shape[0]
        for row in tqdm(issue_description_repr, total=num_rows, desc='rows'):
            issue_description_repr_dense += row.todense().tolist()

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array "
                 f"total time = {(time.time() - tic) / 60} minutes")

        # issue representation.
        log.info(f"Building issue representation ...")
        if self.params['model']['use_structured_data']:
            issue_repr = []
            num_batches = len(issue_description_repr_dense)
            for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
                left_array = issue_description_repr_dense[idx]
                right_array = issue_structured_data[idx]
                left_array.extend(right_array)
                issue_repr.append(left_array)
        else:
            issue_repr = list(issue_description_repr_dense)

        # clear variables to free used memory
        issue_description_repr = None
        issue_structured_data = None

        log.info(f"Building issue representation total time = {(time.time() - tic) / 60} minutes")
        log.info(f"Manual garbage collection: {gc.collect()}")
        # classifier
        log.info(f"Training '{self.model.__class__.__name__}' ...")
        tic = time.time()
        self.model.fit(list(issue_repr), label)
        log.info(f"Training '{self.model.__class__.__name__}' total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Training IssueClassifier total time = {(time.time() - tic) / 60} minutes")

    def predict(self, data, w2i, word_embed):
        # split train data
        issue_description, issue_structured_data, label = DataSplitter().transform(data, self.params['dataset']['task'])

        # TF-IDF features
        log.info(f"Transforming issue description to document-term matrix ...")
        tic = time.time()
        issue_description_repr = self.issue_description_builder.transform(
            issue_description, w2i, word_embed
        )

        log.info(f"Transforming issue description to document-term matrix "
                 f"total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array ...")
        tic = time.time()

        issue_description_repr_dense = []
        num_rows = issue_description_repr.shape[0]
        for row in tqdm(issue_description_repr, total=num_rows, desc='rows'):
            issue_description_repr_dense += row.todense().tolist()

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array "
                 f"total time = {(time.time() - tic) / 60} minutes")

        # issue representation.
        log.info(f"Building issue representation ...")
        if self.params['model']['use_structured_data']:
            issue_repr = []
            num_batches = len(issue_description_repr_dense)
            for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
                left_array = issue_description_repr_dense[idx]
                right_array = issue_structured_data[idx]
                left_array.extend(right_array)
                issue_repr.append(left_array)
        else:
            issue_repr = list(issue_description_repr_dense)

        # clear variables to free used memory
        issue_description_repr = None
        issue_structured_data = None

        log.info(f"Building issue representation total time = {(time.time() - tic) / 60} minutes")

        # clear variables to free used memory
        issue_description_repr = None
        issue_structured_data = None

        log.info(f"Building issue representation total time = {(time.time() - tic) / 60} minutes")

        # classifier prediction
        log.info(f"Obtaining predictions from '{self.model.__class__.__name__}' ...")
        tic = time.time()
        prediction = self.model.predict(list(issue_repr))
        prediction_probability = self.model.predict_proba(list(issue_repr))
        log.info(f"Obtaining predictions from '{self.model.__class__.__name__}' "
                 f"total time = {(time.time() - tic) / 60} minutes")

        return prediction, prediction_probability, label


class IssueSimilarityMeter(CodebooksModel):
    def __init__(
            self,
            params: dict = None,
            model_meta_file: str = None
    ):
        super().__init__(params, model_meta_file)
        self.params = params.copy()

        # Left issue description builder.
        self.issue_description_builder = IssueDescriptionBuilder(self.params['model'])

        # Classifier.
        classifier_name = self.params['model']['classifier']
        classifier_kwargs = get_codebooks_classifier_kwargs(classifier_name, self.params['model'])

        self.model = get_codebooks_classifier(classifier_name)(**classifier_kwargs)

        if model_meta_file is not None and model_meta_file != '':
            self.load_params_model(model_meta_file, 'model', 'model')
            self.load_params_model(model_meta_file, 'issue_description_builder', 'issue_description_builder')

    def fit(self, data, w2i, word_embed):
        log.info(f"Training IssueSimilarityMeter ...")

        # split train data
        issue_description_left, issue_description_right, issue_structured_data_left_repr, \
        issue_structured_data_right_repr, label = DataSplitter().transform(data, self.params['dataset']['task'])

        # TF-IDF features
        log.info(f"Training issue description model and transforming issues descriptions to document-term matrix ...")
        tic = time.time()

        # use left and right issues descriptions
        left_right_issuers_descriptions = issue_description_left + issue_description_right
        issue_description_total_repr = self.issue_description_builder.fit_transform(
            left_right_issuers_descriptions,
            w2i,
            word_embed
        )

        log.info(f"Training issue description model and transforming issue description to document-term matrix "
                 f"total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array ...")
        tic = time.time()

        issue_description_total_repr_dense = []
        num_rows = issue_description_total_repr.shape[0]
        for row in tqdm(issue_description_total_repr, total=num_rows, desc='rows'):
            issue_description_total_repr_dense += row.todense().tolist()

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array "
                 f"total time = {(time.time() - tic) / 60} minutes")

        # split into left and right issues representations
        issue_description_left_repr = issue_description_total_repr_dense[:len(issue_description_left)]
        issue_description_right_repr = issue_description_total_repr_dense[len(issue_description_right):]

        # issue representation.
        log.info(f"Building issue representation ...")
        tic = time.time()
        if self.params['model']['use_structured_data']:
            issue_left_repr = []
            issue_right_repr = []
            num_batches = len(issue_description_left)
            for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
                left_issue_description_left_array = issue_description_left_repr[idx]
                right_issue_structured_data_left_array = issue_structured_data_left_repr[idx]
                left_issue_description_left_array.extend(right_issue_structured_data_left_array)
                issue_left_repr.append(left_issue_description_left_array)

                left_issue_description_right_array = issue_description_right_repr[idx]
                right_issue_structured_data_right_array = issue_structured_data_right_repr[idx]
                left_issue_description_right_array.extend(right_issue_structured_data_right_array)
                issue_right_repr.append(left_issue_description_right_array)

        else:
            issue_left_repr = list(issue_description_left_repr)
            issue_right_repr = list(issue_description_right_repr)

        # clear variables to free used memory
        issue_description_left_repr = None
        issue_structured_data_left_repr = None
        issue_description_right_repr = None
        issue_structured_data_right_repr = None

        log.info(f"Building issue representation total time = {(time.time() - tic) / 60} minutes")

        # Left and right issues representations, and similarity representation.
        log.info(f"Calculating similarity representation ...")
        tic = time.time()
        concatenated_data = []
        num_batches = len(issue_left_repr)
        for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
            subtract_row = [a_i - b_i for a_i, b_i in zip(issue_left_repr[idx], issue_right_repr[idx])]
            abs_row = [abs(ele) for ele in subtract_row]
            subtract_row = None
            multiply_row = [a_i * b_i for a_i, b_i in zip(issue_left_repr[idx], issue_right_repr[idx])]
            abs_row.extend(multiply_row)
            multiply_row = None
            abs_row.extend(issue_left_repr[idx])
            issue_left_repr[idx] = None
            abs_row.extend(issue_right_repr[idx])
            issue_right_repr[idx] = None
            concatenated_data.append(abs_row)

        log.info(f"Calculating similarity representation total time = {(time.time() - tic) / 60} minutes")

        # clear variables to free used memory
        issue_left_repr = None
        issue_right_repr = None

        # classifier
        log.info(f"Training '{self.model.__class__.__name__}' ...")
        tic = time.time()
        self.model.fit(list(concatenated_data), label)
        log.info(f"Training '{self.model.__class__.__name__}' total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Training IssueSimilarityMeter total time = {(time.time() - tic) / 60} minutes")

    def predict(self, data, w2i, word_embed):
        # split train data
        issue_description_left, issue_description_right, issue_structured_data_left_repr, \
        issue_structured_data_right_repr, label = DataSplitter().transform(data, self.params['dataset']['task'])

        # TF-IDF features
        log.info(f"Transforming issue description to document-term matrix ...")
        tic = time.time()
        # use left and right issues descriptions
        left_right_issuers_descriptions = issue_description_left + issue_description_right
        issue_description_total_repr = self.issue_description_builder.transform(
            left_right_issuers_descriptions,
            w2i,
            word_embed
        )

        log.info(f"Training issue description model and transforming issue description to document-term matrix "
                 f"total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array ...")
        tic = time.time()

        issue_description_total_repr_dense = []
        num_rows = issue_description_total_repr.shape[0]
        for row in tqdm(issue_description_total_repr, total=num_rows, desc='rows'):
            issue_description_total_repr_dense += row.todense().tolist()

        log.info(f"Transforming scipy.sparse.csr.csr_matrix to dense array "
                 f"total time = {(time.time() - tic) / 60} minutes")

        # split into left and right issues representations
        issue_description_left_repr = issue_description_total_repr_dense[:len(issue_description_left)]
        issue_description_right_repr = issue_description_total_repr_dense[len(issue_description_right):]

        log.info(f"Transforming issue description to document-term matrix "
                 f"total time = {(time.time() - tic) / 60} minutes")

        # issue representation.
        log.info(f"Building issue representation ...")
        tic = time.time()
        if self.params['model']['use_structured_data']:
            issue_left_repr = []
            issue_right_repr = []
            num_batches = len(issue_description_left)
            for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
                left_issue_description_left_array = issue_description_left_repr[idx]
                right_issue_structured_data_left_array = issue_structured_data_left_repr[idx]
                left_issue_description_left_array.extend(right_issue_structured_data_left_array)
                issue_left_repr.append(left_issue_description_left_array)

                left_issue_description_right_array = issue_description_right_repr[idx]
                right_issue_structured_data_right_array = issue_structured_data_right_repr[idx]
                left_issue_description_right_array.extend(right_issue_structured_data_right_array)
                issue_right_repr.append(left_issue_description_right_array)

        else:
            issue_left_repr = list(issue_description_left_repr)
            issue_right_repr = list(issue_description_right_repr)

        # clear variables to free used memory
        issue_description_left_repr = None
        issue_structured_data_left_repr = None
        issue_description_right_repr = None
        issue_structured_data_right_repr = None

        log.info(f"Building issue representation total time = {(time.time() - tic) / 60} minutes")

        log.info(f"Calculating similarity representation ...")
        tic = time.time()
        # Left and right issues representations, and similarity representation.
        log.info(f"Calculating similarity representation ...")
        tic = time.time()
        concatenated_data = []
        num_batches = len(issue_left_repr)
        for idx in tqdm(range(0, num_batches), total=num_batches, desc='rows'):
            subtract_row = [a_i - b_i for a_i, b_i in zip(issue_left_repr[idx], issue_right_repr[idx])]
            abs_row = [abs(ele) for ele in subtract_row]
            subtract_row = None
            multiply_row = [a_i * b_i for a_i, b_i in zip(issue_left_repr[idx], issue_right_repr[idx])]
            abs_row.extend(multiply_row)
            multiply_row = None
            abs_row.extend(issue_left_repr[idx])
            issue_left_repr[idx] = None
            abs_row.extend(issue_right_repr[idx])
            issue_right_repr[idx] = None
            concatenated_data.append(abs_row)

        log.info(f"Calculating similarity representation total time = {(time.time() - tic) / 60} minutes")

        # clear variables to free used memory
        issue_left_repr = None
        issue_right_repr = None

        # classifier prediction
        log.info(f"Obtaining predictions from '{self.model.__class__.__name__}' ...")
        tic = time.time()
        prediction = self.model.predict(list(concatenated_data))
        prediction_probability = self.model.predict_proba(list(concatenated_data))
        log.info(f"Obtaining predictions from '{self.model.__class__.__name__}' "
                 f"total time = {(time.time() - tic) / 60} minutes")

        return prediction, prediction_probability, label


class IssueAssigner(CodebooksModel):
    def __init__(
            self,
            params: dict = None,
            model_meta_file: str = None
    ):
        super().__init__(params, model_meta_file)
        self.params = params.copy()
        self.active_developers = None
        self.model = None

        if model_meta_file is not None and model_meta_file != '':
            self.load_params_model(model_meta_file, 'model', 'model')

    def _get_normalized_workload(self, train: pd.DataFrame = None) -> pd.DataFrame:
        # workload of each active developer
        df_sum_cost = train.groupby('label', as_index=False)['normalized_cost'].sum()
        df_workload = self.active_developers.merge(df_sum_cost, how='left', on='label')
        df_workload.rename(columns={'normalized_cost': 'sum_normalized_cost'}, errors='raise',
                           inplace=True)
        df_workload['sum_normalized_cost'].fillna(0.0, inplace=True)

        # workload of all active developer
        all_active_developers_workload = df_workload['sum_normalized_cost'].sum() / self.active_developers.shape[0]

        # workload of each active developer in units of their average load
        df_workload['workload'] = df_workload['sum_normalized_cost'] / all_active_developers_workload

        # quantile function of the workload
        workload_list = []
        [workload_list.append([item]) for item in list(df_workload['workload'])]
        p = Pipeline([("a", QuantileTransformer(output_distribution='normal')), ("b", MinMaxScaler())])
        normalized_workload: np.ndarray = p.fit_transform(np.array(workload_list))
        normalized_workload_list = []
        [normalized_workload_list.append(item[0]) for item in list(normalized_workload)]
        data = {'label': list(df_workload['label']), 'normalized_workload': normalized_workload_list}
        df_normalized_workload = pd.DataFrame(data)
        return df_normalized_workload

    def _get_assigned_developers(self, issue_composite_data: list = None):
        issue_global_metric = []
        for index, row in self.model.dropna().iterrows():
            developer_adequacy = []
            for closed_issue in list(row['closed_issues']):
                developer_adequacy.append(jaccard_similarity(issue_composite_data, closed_issue))
            developer_adequacy.sort(reverse=True)
            developer_global_metric = self._get_global_metric(row['normalized_workload'], developer_adequacy[0])
            issue_global_metric.append({'label': row['label'], 'global_metric': developer_global_metric})

        developers = sorted(issue_global_metric, key=lambda k: k['global_metric'], reverse=True)

        return developers[:self.params['model']['k']]

    def _get_global_metric(self, workload, adequacy):
        global_metric = np.float32(
            # self.params['model']['w'] * (1 - workload) + ((1 - self.params['model']['w']) * adequacy)
            self.params['model']['w'] * workload + ((1 - self.params['model']['w']) * adequacy)
        )
        return global_metric

    def fit(self, train: pd.DataFrame = None, opened: pd.DataFrame = None):
        log.info(f"Training IssueAssigner ...")
        datetime_active_developers_str = f"{self.params['model']['active_developer_gte_year']}-" \
                                         f"{self.params['model']['active_developer_gte_month']}-" \
                                         f"{self.params['model']['active_developer_gte_day']}T00:00:00Z"
        datetime_active_developers_filter = datetime.strptime(datetime_active_developers_str, "%Y-%m-%dT%H:%M:%SZ")

        # active developers from train dataset
        train_filtered = train[train['delta_ts'] >= datetime_active_developers_filter].copy()
        train_active_developers = train_filtered['label'].unique()

        # active developers from test dataset
        opened_active_developers = opened['label'].unique()

        # all active developers
        list_all_active_developers = list(train_active_developers) + list(opened_active_developers)
        all_active_developers = set(list_all_active_developers)
        all_active_developers_data = {'label': list(all_active_developers)}
        self.active_developers = pd.DataFrame(all_active_developers_data)

        # add normalized workload
        log.info(f"Adding normalized workload to active developers ...")
        tic = time.time()
        df_active_developers_workload = self._get_normalized_workload(train)
        log.info(f"Adding normalized workload to active developers "
                 f"total execution time = {((time.time() - tic) / 60)} minutes")
        developers_without_workload = df_active_developers_workload[
            df_active_developers_workload['normalized_workload'] == 0.0].copy()
        log.info(f"Number of developers without workload: {developers_without_workload.shape[0]}")

        # developer issues
        log.info(f"Obtaining active developers issues ...")
        tic = time.time()
        developers_issues = get_developer_issues(train)
        log.info(f"Obtaining active developers issues total execution time = {((time.time() - tic) / 60)} minutes")

        log.info(f"Adding active developers closed issues ...")
        tic = time.time()
        df_active_developers_workload['closed_issues'] = df_active_developers_workload['label'].apply(
            lambda x: developers_issues[x] if x in developers_issues.keys() else None)
        log.info(f"Adding active developers closed issues total execution time = {((time.time() - tic) / 60)} minutes")

        self.model = df_active_developers_workload

        log.info(f"Training IssueAssigner total time = {(time.time() - tic) / 60} minutes")

    def predict(self, data: pd.DataFrame = None):
        log.info(f"Obtaining predictions ...")
        tic = time.time()
        y_true = []
        y_prediction = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0], desc='issues'):
            y_true.append(row['label'])
            y_prediction.append(self._get_assigned_developers(row['composite_data']))
        log.info(f"Obtaining predictions total time = {(time.time() - tic) / 60} minutes")

        return y_prediction, y_true


def get_codebooks_model(
        task: str = 'duplicity'
) -> Type[Union[IssueClassifier, IssueSimilarityMeter, IssueAssigner]]:
    codebooks_model = {
        'assignation': IssueClassifier,
        'custom_assignation': IssueAssigner,
        'classification': IssueClassifier,
        'duplicity': IssueSimilarityMeter,
        'prioritization': IssueClassifier,
        'similarity': IssueSimilarityMeter
    }

    return codebooks_model[task]


def _set_kwarg(f, fixed_kwargs) -> object:
    """Closure of a function fixing a kwarg"""

    def f2(*args, **kwargs):
        fixed_kwargs2 = {k: v for k, v in fixed_kwargs.items() if k not in kwargs}
        return f(*args, **fixed_kwargs2, **kwargs)

    return f2


def get_codebooks_classifier(
        classifier: str = 'decision_tree'
) -> Type[Union[DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, LogisticRegression, object]]:
    codebooks_classifier = {
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'extra_trees': ExtraTreesClassifier,
        'logistic_regression': LogisticRegression,
        # 'c_support_vector': _set_kwarg(SVC, {"probability": True})
        'c_support_vector': SVC
    }

    return codebooks_classifier[classifier]


def get_codebooks_classifier_kwargs(
        classifier: str = 'decision_tree',
        params: dict = None
) -> dict:
    tree_kwargs = {}
    for param in ['criterion', 'max_depth', 'max_features', 'random_state', 'min_samples_leaf']:
        if param in params.keys() and params[param] is not None:
            tree_kwargs[param] = params[param]

    linear_kwargs = {}
    for param in ['penalty', 'C', 'multi_class', 'solver']:
        if param in params.keys() and params[param] is not None:
            linear_kwargs[param] = params[param]

    svm_kwargs = {}
    for param in ['C', 'gamma', 'probability']:
        if param in params.keys() and params[param] is not None:
            svm_kwargs[param] = params[param]

    codebooks_classifier_kwargs = {
        'decision_tree': tree_kwargs,
        'random_forest': tree_kwargs,
        'extra_trees': tree_kwargs,
        'logistic_regression': linear_kwargs,
        'c_support_vector': svm_kwargs
    }

    return codebooks_classifier_kwargs[classifier]
