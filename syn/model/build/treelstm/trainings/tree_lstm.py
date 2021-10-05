import os
import pickle
from pathlib import Path

import dynet as dy
import log4p
from dotenv import load_dotenv

from definitions import ROOT_DIR, SYN_ENV
from syn.model.build.codebooks.incidences import data
from syn.model.build.codebooks.incidences.tasks import Task
from syn.model.build.treelstm.tree import TreeLstm, TreeLstmCategorical
from syn.model.build.treelstm.vectorizer.Vectorizer import get_vectorized_issue

env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
load_dotenv(dotenv_path=env_path)

# Define el logger que se utilizará.
logger = log4p.GetLogger(__name__)
log = logger.logger


class TreeLstmDuplicateTrain(Task):
    """The task of training a duplicate detector"""

    def __init__(
            self,
            corpus='bugzilla',
            collection='clear',
            architecture='LSTM',
            attention=True,
            attention_size=10,
            glove_size=100,
            hidden_size=100,
            max_input=200,
            batch_size=1,
            optimizer='ADAM',
            learning_rate=0.001,
            update_embeddings=True,
            patience=5
    ):

        super().__init__()
        self.solution_name = 'tree_lstm'
        self.kwargs = {
            'corpus': corpus,
            'collection': collection,
            'attention_size': attention_size,
            'glove_size': glove_size,
            'hidden_size': hidden_size,
            'max_input': max_input,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'update_embeddings': update_embeddings,
            'patience': patience,
            'attention': attention,
        }

        self.model_dump_path = Path(os.environ.get('MODEL_DUMP_PATH')) / self.solution_name / self.get_file_name()

        # self.data = data.create_embeddings_tree(vectorized_issue, input_vocab, self.model_dump_path)
        #
        # self.model = TreeLstm(
        #     dy.Model(),
        #     data_train,
        #     data_test,
        #     embeddings,
        #     self.model_dump_path,
        #     update_embeddings=self.kwargs["update_embeddings"],
        #     hidden_dim=self.kwargs["hidden_size"],
        #     attention_size=self.kwargs["attention_size"],
        #     batch_size=self.kwargs["batch_size"],
        #     learning_rate=self.kwargs["learning_rate"],
        #     patience=self.kwargs["patience"],
        #     attention=self.kwargs["attention"],
        #     corpus=self.kwargs["corpus"]
        # )

    def run(self):
        # Issue vectorization.
        vectorized_issue = get_vectorized_issue(self.kwargs["corpus"], self.kwargs["collection"],
                                                self.kwargs["glove_size"])

        # Training dataset from vectorized issues.
        raw_data = vectorized_issue.attention_vector_raw_data

        data_train, data_test, input_vocab = data.get_dataset_tree(
            raw_data,
            max_sentence_length=self.kwargs["max_input"]
        )

        # Word embeddings.
        embeddings = data.create_embeddings_tree(vectorized_issue, input_vocab, self.model_dump_path)

        model = dy.Model()

        tree_model = TreeLstm(
            model,
            data_train,
            data_test,
            embeddings,
            self.model_dump_path,
            update_embeddings=self.kwargs["update_embeddings"],
            hidden_dim=self.kwargs["hidden_size"],
            attention_size=self.kwargs["attention_size"],
            batch_size=self.kwargs["batch_size"],
            learning_rate=self.kwargs["learning_rate"],
            patience=self.kwargs["patience"],
            attention=self.kwargs["attention"],
            corpus=self.kwargs["corpus"]
        )

        tree_model.fit()
        self.results = tree_model.evaluate()
        return self.results

    def load_model(self, model_name):
        # Issue vectorization.
        vectorized_issue = get_vectorized_issue(self.kwargs["corpus"], self.kwargs["collection"],
                                                self.kwargs["glove_size"])

        # Training dataset from vectorized issues.
        raw_data = vectorized_issue.attention_vector_raw_data

        data_train, data_test, input_vocab = data.get_dataset_tree(
            raw_data,
            max_sentence_length=self.kwargs["max_input"]
        )

        # Word embeddings.
        embeddings = data.create_embeddings_tree(vectorized_issue, input_vocab, self.model_dump_path)
        model = dy.Model()
        data_train = None
        data_test = None
        model_tree = TreeLstm(
            model,
            data_train,
            data_test,
            embeddings,
            self.model_dump_path,
            update_embeddings=self.kwargs["update_embeddings"],
            hidden_dim=self.kwargs["hidden_size"],
            attention_size=self.kwargs["attention_size"],
            batch_size=self.kwargs["batch_size"],
            learning_rate=self.kwargs["learning_rate"],
            patience=self.kwargs["patience"],
            attention=self.kwargs["attention"]
        )
        if os.path.exists(self.model_dump_path + "input_vocab.txt"):
            os.remove(self.model_dump_path + "input_vocab.txt")
        if os.path.exists(self.model_dump_path + "input_embeddings.npy"):
            os.remove(self.model_dump_path + "input_embeddings.npy")
        model_tree.model.populate(self.model_dump_path)
        return model_tree


class TreeLstmCategoricalTrain(Task):
    """The task of training a duplicate detector"""

    def __init__(self, corpus, collection, attention=True, attention_size=10,
                 glove_size=100, hidden_size=100, max_input=200, batch_size=1, optimizer='ADAM', learning_rate=0.001,
                 update_embeddings=True, patience=5, column='priority', num_samples=-1, train_porcent=0.8,
                 balanced=False, num_cat=2, attention_vector=True):

        super().__init__()
        self.kwargs = {
            "corpus": corpus,
            "collection": collection,
            "attention_size": attention_size,
            "glove_size": glove_size,
            "hidden_size": hidden_size,
            "max_input": max_input,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "update_embeddings": update_embeddings,
            "patience": patience,
            "attention": attention,
            "column": column,
            "num_samples": num_samples,
            "train_porcent": train_porcent,
            "balanced": balanced,
            "num_cat": num_cat,
            "attention_vector": attention_vector,
        }

    def run(self, model_name):

        train_name = os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(
            extension='')

        # Codifica las incidencias.
        vectorized_issue = get_vectorized_issue(self.kwargs["corpus"], self.kwargs["collection"],
                                                self.kwargs["glove_size"],
                                                attention_vector=self.kwargs["attention_vector"], categorical=True,
                                                column=self.kwargs["column"])

        # Generación de los conjuntos de entrenamiento a partir de las incidencias codificadas.
        raw_data = vectorized_issue.attention_vector_raw_data

        raw_data_post = data.get_dataset_other_categorical(raw_data, self.kwargs["num_samples"],
                                                           self.kwargs["balanced"],
                                                           self.kwargs["num_cat"], self.kwargs["column"])
        data_train, data_test, input_vocab = data.get_dataset_tree_categorical(raw_data_post,
                                                                               max_sentence_length=self.kwargs[
                                                                                   "max_input"],
                                                                               column=self.kwargs["column"],
                                                                               train_size=self.kwargs["train_porcent"]
                                                                               )

        embeddings = data.create_embeddings_tree(vectorized_issue, input_vocab, train_name)

        model = dy.Model()
        model_tree = TreeLstmCategorical(
            model,
            data_train,
            data_test,
            embeddings,
            train_name,
            update_embeddings=self.kwargs["update_embeddings"],
            hidden_dim=self.kwargs["hidden_size"],
            attention_size=self.kwargs["attention_size"],
            batch_size=self.kwargs["batch_size"],
            learning_rate=self.kwargs["learning_rate"],
            patience=self.kwargs["patience"],
            attention=self.kwargs["attention"],
            corpus=self.kwargs["corpus"],
            num_cat=self.kwargs["num_cat"],
        )

        model_tree.fit()
        self.results = model_tree.evaluate()
        return self.results

    def run_and_pickle(self, model_name='codebooks', remote_base_path=None):
        """Execute the task and store the results as a pickle and in the db"""
        result = self.run(model_name)
        pickle.dump(result,
                    open(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(),
                         "wb"))
        self.put_sftp(
            os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(extension=''),
            remote_base_path)
        self.put_sftp(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(),
                      remote_base_path)
        self._db_store()
        return self.load_model(model_name), result

    def load_or_run(self, model_name='codebooks', remote_base_path=None):
        """Load the results if available, otherwise running the task, storing the results, and returning them"""
        try:
            model = self.load_model(model_name)
        except RuntimeError:
            try:
                self.get_sftp(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(
                    extension=''), remote_base_path)
                self.get_sftp(
                    os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(),
                    remote_base_path)
                return self.load_model(model_name), self.unpickle(model_name)
            except FileNotFoundError:
                return self.run_and_pickle(model_name, remote_base_path)
        return model, self.unpickle(model_name)

    def unpickle(self, model_name):
        """Load the results of the task if available"""
        self._db_store()
        return pickle.load(
            open(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(), "rb"))

    def delete(self, model_name='codebooks', remote_base_path=None):
        os.remove(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name())
        os.remove(
            os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(extension=''))
        self.remove_sftp(os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(),
                         remote_base_path)
        self.remove_sftp(
            os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(extension=''),
            remote_base_path)

    def load_model(self, model_name):

        train_name = os.environ.get('MODEL_DUMP_PATH') + self.get_file_name(
            extension='')

        # Codifica las incidencias.
        vectorized_issue = get_vectorized_issue(self.kwargs["corpus"], self.kwargs["collection"],
                                                self.kwargs["glove_size"],
                                                attention_vector=self.kwargs["attention_vector"], categorical=True,
                                                column=self.kwargs["column"])

        # Generación de los conjuntos de entrenamiento a partir de las incidencias codificadas.
        raw_data = vectorized_issue.attention_vector_raw_data

        raw_data_post = data.get_dataset_other_categorical(raw_data, self.kwargs["num_samples"],
                                                           self.kwargs["balanced"],
                                                           self.kwargs["num_cat"], self.kwargs["column"])
        data_train, data_test, input_vocab = data.get_dataset_tree_categorical(raw_data_post,
                                                                               max_sentence_length=self.kwargs[
                                                                                   "max_input"],
                                                                               column=self.kwargs["column"],
                                                                               train_size=self.kwargs["train_porcent"]
                                                                               )

        # Se obtienen los embeddings
        embeddings = data.create_embeddings_tree(vectorized_issue, input_vocab, train_name)
        model = dy.Model()
        data_train = None
        data_test = None
        model_tree = TreeLstmCategorical(
            model,
            data_train,
            data_test,
            embeddings,
            train_name,
            update_embeddings=self.kwargs["update_embeddings"],
            hidden_dim=self.kwargs["hidden_size"],
            attention_size=self.kwargs["attention_size"],
            batch_size=self.kwargs["batch_size"],
            learning_rate=self.kwargs["learning_rate"],
            patience=self.kwargs["patience"],
            attention=self.kwargs["attention"],
            num_cat=self.kwargs["num_cat"]
        )
        if os.path.exists(train_name + "input_vocab.txt"):
            os.remove(train_name + "input_vocab.txt")
        if os.path.exists(train_name + "input_embeddings.npy"):
            os.remove(train_name + "input_embeddings.npy")
        print(train_name)
        model_tree.model.populate(train_name)
        return model_tree
