import argparse
from argparse import ArgumentParser

# argparse.ArgumentError: argument -h/--help: conflicting option strings: -h, --help

# Parameters for operations of type 'all'.
all_operations_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
all_operations_parser.add_argument('--task_list', default='', help='Task name list.')
all_operations_parser.add_argument('--corpus_list', default='', help='Corpus name list.')
all_operations_parser.add_argument('--embeddings_model_list', default='', help='Embeddings model list.')
all_operations_parser.add_argument('--embeddings_size_list', default='', help='Embeddings size list.')

# MongoDB parameters
mongodb_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)

mongodb_parser.add_argument("--mh", "--mongodb_host", default="localhost", type=str,
                            choices=['localhost', 'syn.altgovrd.com'], help="MongoDB server.")
mongodb_parser.add_argument("--mp", "--mongodb_port", default=30017, type=int, help="MongoDB port.")
mongodb_parser.add_argument("--db", default="test", type=str, help="MongoDB database name.")
mongodb_parser.add_argument("--c", default="test", type=str, help="MongoDB collection name.")
mongodb_parser.add_argument("--pj", default=None, type=str, help="Projection fields separated by commas.")
mongodb_parser.add_argument("--ql", default=0, type=int, help="MongoDB query limit (0 no limit).")
mongodb_parser.add_argument("--dc", default=True, action='store_true', help="Drop collection if exists.")

# Common parameters
common_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--database_name', default='tasks', type=str, help="Database name.")
common_parser.add_argument('--collection_name', default='experiments', type=str, help="Collection name.")
common_parser.add_argument('--query_limit', default=0, type=int, help='Results size (zero for all results).')
common_parser.add_argument('--corpus', default='bugzilla', type=str,
                           choices=['bugzilla', 'eclipse', 'netBeans', 'openOffice', 'gerrit'], help='Corpus.')
common_parser.add_argument('--mongo_batch_size', default=10000, type=int, help='Batch size.')

# Vocabulary parameters
vocabulary_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
vocabulary_parser.add_argument('--vocabulary_name', default='vocabulary', type=str,
                               help='Vocabulary collection name.')
vocabulary_parser.add_argument('--tokens_collection', default='normalized_clear', type=str,
                               help='Tokens collection name.')

# Task parameters
task_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
task_parser.add_argument('--alias', default='base_params', help='Task alias')

# Dataset parameters
dataset_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument('--task', default='duplicity', type=str,
                            choices=['assignation', 'custom_assignation', 'classification', 'duplicity',
                                     'prioritization', 'similarity'], help='Task for which the dataset is built.')
dataset_parser.add_argument('--corpus', default='bugzilla', type=str,
                            choices=['bugzilla', 'eclipse', 'netBeans', 'openOffice', 'gerrit'],
                            help='Corpus for which the dataset is built.'
                            )
dataset_parser.add_argument('--dataset_name', default='normalized_clear', type=str, help='Collection name or filename.')
dataset_parser.add_argument('--no_balance_data', default=False, dest='balance_data', action='store_false',
                            help="No balance data.")
dataset_parser.add_argument('--balance_data', dest='balance_data', action='store_true', help="Balance data.")
dataset_parser.add_argument('--query_limit', default=0, type=int, help='Dataset size (zero for the full data set).')

# Source parameters
source_params_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
source_params_parser.add_argument('--params_file', default='', type=str, help="JSON params file.")
source_params_parser.add_argument('--database_name', default='tasks', type=str, help="Params database.")
source_params_parser.add_argument('--collection_name', default='experiments', type=str, help="Params collection.")
source_params_parser.add_argument('--params_id', default='', type=str, help="Params identifier.")
source_params_parser.add_argument('--no_reference_params', default=False, dest='reference_params', action='store_false',
                                  help="No reference params.")
source_params_parser.add_argument('--reference_params', dest='reference_params', action='store_true',
                                  help="Reference params.")
source_params_parser.add_argument('--model_meta_file', default='', type=str, help="Model meta file.")

# hyper search parameters
hyper_search_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
hyper_search_parser.add_argument('--hyper_search_objective', default='accuracy', type=str,
                                 help="Hyper search objective.")
hyper_search_parser.add_argument('--no_hyper_search_overwrite', default=False, dest='hyper_search_overwrite',
                                 action='store_false', help="No overwrite previous hyper searches.")
hyper_search_parser.add_argument('--hyper_search_overwrite', dest='hyper_search_overwrite', action='store_true',
                                 help="Overwrite previous hyper searches.")

# scheduler parameters
scheduler_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
scheduler_parser.add_argument('--trainer', default='AdagradTrainer', help='Trainer name in dynet')
scheduler_parser.add_argument('--sparse', default=1, type=int, help='Sparse update 0/1')
scheduler_parser.add_argument('--learning_rate_param', default=0.05, type=float)
scheduler_parser.add_argument('--learning_rate_embed', default=0.005, type=float)
scheduler_parser.add_argument('--model_save_dir', default='')
scheduler_parser.add_argument('--batch_size', default=25, type=int)
scheduler_parser.add_argument('--loss_function', default='cross_entropy_loss', type=str,
                              choices=['cross_entropy_loss', 'hinge_loss'])
scheduler_parser.add_argument('--regularization_strength', default=1e-4, type=float)
scheduler_parser.add_argument('--max_epochs', default=100, type=int, help='Number of epochs to train')

# classifier parameters
classifier_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
classifier_parser.add_argument('--n_classes', default=None, type=float)
classifier_parser.add_argument('--num_layers', default=None, type=int, help='biLSTM number of layers')
classifier_parser.add_argument('--min_df', default=1, type=int, help='TF-IDF parameter')
classifier_parser.add_argument('--criterion', default=None, type=str, help='Tree classifier parameter')
classifier_parser.add_argument('--max_depth', default=None, type=int, help='Tree classifier parameter')
classifier_parser.add_argument('--max_features', default=None, type=int, help='Tree classifier parameter')
classifier_parser.add_argument('--random_state', default=None, type=int, help='Common classifier parameter')
classifier_parser.add_argument('--min_samples_leaf', default=None, type=int, help='Linear classifier parameter')
classifier_parser.add_argument('--penalty', default=None, type=str, help='Linear classifier parameter')
classifier_parser.add_argument('--c', default=None, type=float, help='Linear classifier parameter')
classifier_parser.add_argument('--multi_class', default=None, type=str, help='SVM classifier parameter')
classifier_parser.add_argument('--solver', default='liblinear', type=str, help='')
classifier_parser.add_argument('--gamma', default=None, type=str, help='SVM classifier parameter')
classifier_parser.add_argument('--probability', default=True, dest='probability', action='store_true',
                               help="SVM classifier parameter")
classifier_parser.add_argument('--no_probability', dest='probability', action='store_false',
                               help="SVM classifier parameter")

# sentence model parameters
sentence_model_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
sentence_model_parser.add_argument('--architecture', default='tree_lstm', type=str,
                                   choices=['bi_lstm', 'tree_lstm', 'codebooks'], help="Model architecture.")
sentence_model_parser.add_argument('--sentence_hidden_dim', default=300, type=int, help='sentence hidden dimension')
sentence_model_parser.add_argument('--attention', default='none', type=str, choices=['none', 'leaf', 'root'])
sentence_model_parser.add_argument('--attention_dim', default=32, type=int)

# Word embeddings parameters
embeddings_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
embeddings_parser.add_argument('--embeddings_size', default=300, type=int, choices=[100, 300],
                               help='Word embeddings size.')
embeddings_parser.add_argument('--embeddings_model', default='glove', type=str,
                               choices=['glove', 'word2vec', 'fasttext'], help='Word embeddings model.')
embeddings_parser.add_argument('--embeddings_pretrained', default=True, dest='embeddings_pretrained',
                               action='store_true', help="Pre-trained word embeddings.")
embeddings_parser.add_argument('--no_embeddings_pretrained', dest='embeddings_pretrained', action='store_false',
                               help="Untrained word embeddings.")

# structured data model parameters
structured_data_model_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
structured_data_model_parser.add_argument('--structured_data_input_dim', default=3, type=int,
                                          help="Structured data input dimension.")
structured_data_model_parser.add_argument('--structured_data_num_layers', default=2, type=int,
                                          help="Structured data num layers.")
structured_data_model_parser.add_argument('--structured_data_hidden_dim', default=512, type=int,
                                          help='Structured data hidden dimension')
structured_data_model_parser.add_argument('--structured_data_dropout_rate', default=0.4, type=float,
                                          help='Structured data dropout rate')
structured_data_model_parser.add_argument('--use_structured_data', default=True, dest='use_structured_data',
                                          action='store_true', help="Use structured data.")
structured_data_model_parser.add_argument('--no_use_structured_data', dest='use_structured_data', action='store_false',
                                          help="No use structured data.")

# Assignation task parameters
assignation_task_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
assignation_task_parser.add_argument('--n_developers', default=30, help='Number of developers')

# Similarity task parameters
similarity_task_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
similarity_task_parser.add_argument('--no_near_issues', default=False, dest='near_issues', action='store_false',
                                    help="No near issues.")
similarity_task_parser.add_argument('--near_issues', dest='near_issues', action='store_true', help="Near issues.")

# DyNet global settings
dynet_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
dynet_parser.add_argument("--dynet-seed", default=0, type=int)
dynet_parser.add_argument("--dynet-mem", default=512, type=int)
dynet_parser.add_argument("--dynet-gpus", default=0, type=int)
dynet_parser.add_argument("--dynet-autobatch", default=0, type=int)

# Codebooks parameters
codebooks_parser: ArgumentParser = argparse.ArgumentParser(add_help=False)
codebooks_parser.add_argument('--n_codewords', default=50, type=int, help='Number of codewords (clusters).')
codebooks_parser.add_argument('--list_n_codewords', default='', type=str, help='Number of codewords (clusters) list.')
codebooks_parser.add_argument('--classifier', default='decision_tree', type=str,
                              choices=['decision_tree', 'random_forest', 'extra_trees', 'logistic_regression',
                                       'c_support_vector'], help="Classifier.")
codebooks_parser.add_argument('--structured_data_input_dim', default=3, type=int,
                              help="Structured data input dimension.")
codebooks_parser.add_argument('--use_structured_data', default=True, dest='use_structured_data',
                              action='store_true', help="Use structured data.")
codebooks_parser.add_argument('--no_use_structured_data', dest='use_structured_data', action='store_false',
                              help="No use structured data.")
codebooks_parser.add_argument('--model_save_dir', default='', help='Directory to save model.')
codebooks_parser.add_argument('--parent_class', default='train', type=str, help='')
