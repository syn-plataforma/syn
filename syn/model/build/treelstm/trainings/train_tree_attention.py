import time
from pathlib import Path

import dynet as dy
import dynet_config
import log4p
from dotenv import load_dotenv

from definitions import ROOT_DIR, SYN_ENV
from syn.model.build.treelstm.vectorizer.Vectorizer import get_vectorized_issue
from syn.model.build.treelstm.trainings.attention_tag import get_attention, plot_attention
from syn.model.build.treelstm.trainings.models import Classifier, Tree_Lstm

env_path = Path(ROOT_DIR) / 'config' / (SYN_ENV + '.env')
load_dotenv(dotenv_path=env_path)

# Defines logger.
logger = log4p.GetLogger(__name__)
log = logger.logger

MEMORIA_ASIGNADA = 14000
dynet_config.set(mem=MEMORIA_ASIGNADA, requested_gpus=1, autobatch=True)
dynet_config.set_gpu()


def train_model(
        net,
        model_name,
        max_sentence_length,
        attention_size,
        data
):

    log.info(f"Maximum sentence length: '{str(max_sentence_length)}'.")
    _data_train = [(l, s1, s2, t1, t2, ls1, ls2) for _, l, s1, s2, t1, t2, ls1, ls2 in data[0] if
                   len(ls1) <= max_sentence_length and len(ls1) > 1 and len(ls2) <= max_sentence_length and len(
                       ls2) > 1]
    _data_test = [(l, s1, s2, t1, t2, ls1, ls2) for _, l, s1, s2, t1, t2, ls1, ls2 in data[1] if
                  len(ls1) <= max_sentence_length and len(ls1) > 1 and len(ls2) <= max_sentence_length and len(ls2) > 1]

    l = list(_data_train)
    random.shuffle(l)
    _data_train = tuple(l)
    l = list(_data_test)
    random.shuffle(l)
    _data_test = tuple(l)

    num_batches = len(_data_train)
    print("Training data contains " + str(num_batches))
    NUMBER_TAG = 2
    classifier = Classifier(model, 100 + NUMBER_TAG)

    trainer = dy.AdamTrainer(model, float(LEARNING_RATE))

    validate_frequency = num_batches // 5

    report_frequency = 500 * 16

    start_time = time()
    last_validated = None
    last_reported = None
    best_validation = 0
    validations = []
    validation_means = []
    avg_window_size = 5
    patience = 4
    frustration = 0
    early_stop = False
    epoch = 0
    batches_seen = 0

    att = get_attention('att_tag', net.hidden_dim, NUMBER_TAG,
                        attention_size,
                        model)

    start = time()

    while True:
        print("Start of epoch #" + str(epoch))
        for batch_num, data in enumerate(_data_train):
            ls, s1, s2, t1, t2, ls1, ls2 = data
            if str(s1) != '()' and str(s2) != '()':
                dy.renew_cg()
                out_tree_1 = net.do_parse_tree(s1, len(t1))
                temp = pd.DataFrame(t1).values
                matrix_tag = dy.inputTensor(temp)
                matrix_tag = dy.transpose(matrix_tag)
                out_tree_attention_1, pesos_1 = att(out_tree_1.c, matrix_tag, False)
                out_tree_2 = net.do_parse_tree(s2, len(t2))
                temp2 = pd.DataFrame(t2).values
                matrix_tag = dy.inputTensor(temp2)
                matrix_tag = dy.transpose(matrix_tag)
                out_tree_attention_2, pesos_2 = att(out_tree_2.c, matrix_tag, False)
                predicted_labels = classifier(dy.concatenate([out_tree_1.h, out_tree_attention_1]),
                                              dy.concatenate([out_tree_2.h, out_tree_attention_2]))
                loss = dy.pickneglogsoftmax(predicted_labels, ls)

                # optimise
                loss.forward()
                loss.backward()
                trainer.update()

                if batches_seen % validate_frequency == 0 and last_validated != batches_seen:
                    last_validated = batches_seen

                    acc, _, _, _, _ = eval_dataset(net, classifier, _data_test, att)
                    validations.append(acc)
                    validation_means.append(np.mean(validations[-avg_window_size:]))
                    print("Validation: accuracy " + str(acc) + ", moving average " + str(validation_means[-1]))
                    if acc >= best_validation:
                        best_validation = acc
                        if os.path.exists(DATA_PATH + model_name):
                            os.remove(DATA_PATH + model_name)
                        model.save(DATA_PATH + model_name)
                        print("Modelo guardado. . . ")
                        frustration = 0

                    if len(validation_means) > patience and validation_means[-1] <= np.array(
                            validation_means[:-patience]).max():
                        frustration += 1
                        if frustration > patience:
                            print("Parada temprana. . . ")
                            early_stop = True
                            break
                    else:
                        frustration = 0

                # Report progress
                if batches_seen % report_frequency == 0 and last_reported != batches_seen:
                    last_reported = batches_seen
                    fraction_done = batch_num / num_batches

                    # Update temperature
                    if isinstance(net, Tree_Lstm):
                        net.inv_temp = (float(epoch) + fraction_done) * 100.0 + 1.0
                batches_seen += 1
        if early_stop:
            break
        epoch += 1
    print("Entrenamiento " + str(model_name) + " finalizado.")

    final = time()
    print("Tiempo del entrenamiento: ")
    print(final - start)
    acc, pre, rec, f1, acc2 = eval_dataset(net, classifier, _data_test, att)
    print("Accuracy: " + str(acc))
    print("Precision: " + str(pre))
    print("Recall: " + str(rec))
    print("F1_score: " + str(f1))
    print("Tiempo evaluación del modelo (min): " + str((time() - start_time) / 60))
    print_att(net, classifier, _data_test, att)


def print_att(net, classifier, dataset, att):
    for l, s1, s2, t1, t2, ls1, ls2 in dataset:
        dy.renew_cg()
        out_tree_1 = net.do_parse_tree(s1, len(t1))
        temp = pd.DataFrame(t1).values
        matrix_tag = dy.inputTensor(temp)
        matrix_tag = dy.transpose(matrix_tag)
        out_tree_attention_1, pesos_1 = att(out_tree_1.c, matrix_tag, False)
        out_tree_2 = net.do_parse_tree(s2, len(t2))
        temp = pd.DataFrame(t2).values
        matrix_tag = dy.inputTensor(temp)
        matrix_tag = dy.transpose(matrix_tag)
        out_tree_attention_2, pesos_2 = att(out_tree_2.c, matrix_tag, False)
        predicted = classifier(dy.concatenate([out_tree_1.h, out_tree_attention_1]),
                               dy.concatenate([out_tree_2.h, out_tree_attention_2])).tensor_value().argmax().as_numpy()
        plot_attention(pesos_1, ls1, pesos_2, ls2, predicted, l)


def eval_dataset(net, classifier, dataset, att):
    accurate = 0.0
    TP_rate, TN_rate, FP_rate, FN_rate = 0, 0, 0, 0
    total = 0.0
    for l, s1, s2, t1, t2, ls1, ls2 in dataset:
        dy.renew_cg()
        out_tree_1 = net.do_parse_tree(s1, len(t1))
        temp = pd.DataFrame(t1).values
        matrix_tag = dy.inputTensor(temp)
        matrix_tag = dy.transpose(matrix_tag)
        out_tree_attention_1, pesos_1 = att(out_tree_1.c, matrix_tag, False)
        out_tree_2 = net.do_parse_tree(s2, len(t2))
        temp = pd.DataFrame(t2).values
        matrix_tag = dy.inputTensor(temp)
        matrix_tag = dy.transpose(matrix_tag)
        out_tree_attention_2, pesos_2 = att(out_tree_2.c, matrix_tag, False)
        predicted = classifier(dy.concatenate([out_tree_1.h, out_tree_attention_1]),
                               dy.concatenate([out_tree_2.h, out_tree_attention_2])).tensor_value().argmax().as_numpy()
        r = np.sum(np.equal(l, predicted))
        if l == 0:
            TP = np.sum(np.equal(l, predicted))
            FP = np.sum(np.not_equal(l, predicted))
            FP_rate += FP
            TP_rate += TP
        if l == 1:
            TN = np.sum(np.equal(l, predicted))
            FN = np.sum(np.not_equal(l, predicted))
            FN_rate += FN
            TN_rate += TN
        accurate += r
        total = total + 1.0
    print(TP_rate)
    print(TN_rate)
    print(FN_rate)
    print(FP_rate)
    precision = TP_rate / (TP_rate + FP_rate)
    recall = TP_rate / (TP_rate + FN_rate)
    accuracy = (TP_rate + TN_rate) / (TP_rate + TN_rate + FP_rate + FN_rate)
    f1_score = (2 * precision * recall) / (precision + recall)
    return accurate / total, precision, recall, f1_score, accuracy


if __name__ == "__main__":

    import random
    import os
    import pandas as pd
    from time import time
    import numpy as np

    from syn.model.common.utils_sgd import OTFVocab, parse_ptb_sexpr

    ATTENTION_SIZE = 25
    TRAIN_NAME = 'description_attention' + str(ATTENTION_SIZE)
    GLOVE_SIZE = 100
    DIMENSION_INPUT = 200
    DIMENSION_HIDDEN = 100

    label_map = {
        1: 0,
        -1: 1
    }

    tiempo_inicio = time()
    if not os.path.exists('./resultados/'):
        os.makedirs('./resultados/')
    output_dir = './resultados/' + TRAIN_NAME + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Codifica las incidencias.
    vectorized_issue = get_vectorized_issue("eclipse", "new_eclipse_duplicate_det_task", GLOVE_SIZE)

    input_vocab = OTFVocab()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    start = time()

    # Generación de los conjuntos de entrenamiento a partir de las incidencias codificadas.
    data = vectorized_issue.attention_vector_raw_data

    lista_dict_data = list(data)
    df = pd.DataFrame(lista_dict_data)
    df['dec'] = df['dec'].apply(lambda x: label_map[x])
    df_duplicate = df.loc[df["dec"] == 0]
    df_no_duplicate = df.loc[df["dec"] == 1]
    limit = min(df_duplicate.shape[0], df_no_duplicate.shape[0])
    df_duplicate = df_duplicate.head(limit)
    df_no_duplicate = df_no_duplicate.head(limit)
    train_dup_size = round(0.8 * df_duplicate.shape[0])
    test_dup_size = round(0.2 * df_duplicate.shape[0])
    train_nodup_size = round(0.8 * df_no_duplicate.shape[0])
    test_nodup_size = round(0.2 * df_no_duplicate.shape[0])
    df_train_dup = df_duplicate.iloc[0:train_dup_size]
    df_test_dup = df_duplicate.iloc[train_dup_size:train_dup_size + test_dup_size]
    df_train_nodup = df_no_duplicate.iloc[0:train_nodup_size]
    df_test_nodup = df_no_duplicate.iloc[train_nodup_size:train_nodup_size + test_nodup_size]
    df_train = df_train_dup.append(df_train_nodup)
    df_test = df_test_dup.append(df_test_nodup)
    df_train["constituency_tree_description1"] = df_train["constituency_tree_description1"].apply(
        lambda x: parse_ptb_sexpr(str(x), input_vocab))
    df_train["constituency_tree_description2"] = df_train["constituency_tree_description2"].apply(
        lambda x: parse_ptb_sexpr(str(x), input_vocab))
    df_test["constituency_tree_description1"] = df_test["constituency_tree_description1"].apply(
        lambda x: parse_ptb_sexpr(str(x), input_vocab))
    df_test["constituency_tree_description2"] = df_test["constituency_tree_description2"].apply(
        lambda x: parse_ptb_sexpr(str(x), input_vocab))
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print('Balanceo de datos train: ')
    print(df_train['dec'].value_counts())
    print('Balanceo de datos evaluation: ')
    print(df_test['dec'].value_counts())

    records_train = df_train.to_records(index=False)
    result_train = list(records_train)
    print("Tamaño del entrenamiento train: ")
    print(len(result_train))
    records_test = df_test.to_records(index=False)
    result_test = list(records_test)
    print("Tiempo del preprocesamiento: ")
    final = time()
    print(final - start)

    # calculate UNK by averaging the embeddings we do have in the vocab
    unk = np.zeros((vectorized_issue.pretrained_embeddings.embeddings.shape[1],))
    in_vocab = 0
    for word in input_vocab.vocab:
        if word in vectorized_issue.pretrained_embeddings.reveresed_vocabulary:
            unk += vectorized_issue.pretrained_embeddings.embeddings[
                vectorized_issue.pretrained_embeddings.reveresed_vocabulary[word]]
            in_vocab += 1
    unk = unk / float(in_vocab)
    print("Set " + str(len(input_vocab) - in_vocab) + "/" + str(
        len(input_vocab)) + " missing words in the pretrained embeddings to UNK")

    # save the mutable embeddings and the vocab
    embeddings = []
    if os.path.exists(output_dir + "input_vocab.txt"):
        os.remove(output_dir + "input_vocab.txt")
    with open(output_dir + "input_vocab.txt", "w", encoding='utf-8') as fout:
        for word in input_vocab.vocab:
            if word in vectorized_issue.pretrained_embeddings.reveresed_vocabulary:
                embeddings.append(vectorized_issue.pretrained_embeddings.embeddings[
                                      vectorized_issue.pretrained_embeddings.reveresed_vocabulary[word]])
            else:
                embeddings.append(unk)
            fout.write(word + "\n")
    embeddings = np.array(embeddings, dtype=np.float32)

    DATA_PATH = "resultados/" + TRAIN_NAME + "/"
    BATCH_SIZE = 1
    OPTIMIZER = 'ADAM'
    LEARNING_RATE = 0.001
    model = dy.Model()
    restart = None
    update_embeddings = True
    dir_path = os.path.dirname(os.path.realpath(__file__))

    DIMENSION_GLOVE = embeddings.shape[1]
    ATTENTION_NAME = 'att_tag'
    net = Tree_Lstm(
        model,
        embeddings,
        update_embeddings=update_embeddings,
        hidden_dim=DIMENSION_HIDDEN,
    )

    train_model(
        net,
        model_name=TRAIN_NAME,
        max_sentence_length=DIMENSION_INPUT,
        attention_size=ATTENTION_SIZE,
        data=(result_train, result_test),
    )
