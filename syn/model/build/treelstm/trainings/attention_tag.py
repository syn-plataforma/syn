from __future__ import print_function, division

import dynet as dy

import sys
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
from numpy import array
sys.setrecursionlimit(10000000)

class Attention(object):
    """docstring for Attention"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('att')

    def init(self, test=True, update=True):
        pass

    def __call__(self, H, h, test=True):
        raise NotImplemented()

class Attention_Tag(Attention):
    """docstring for MLPAttention"""

    def __init__(self, di, dh, da, pc):
        super(Attention_Tag, self).__init__(pc)
        self.di, self.dh, self.da = di, dh, da
        # Parameters
        self.W1 = self.pc.add_parameters((self.da), name='W1')
        self.W3 = self.pc.add_parameters((self.da, self.di), name='W3')
        self.W2 = self.pc.add_parameters((self.da, self.dh), name='W2')

    def __call__(self, out_tree, matrix_tag, test=True):
        W3, W2, W1 = dy.parameter(self.W3), dy.parameter(self.W2), dy.parameter(self.W1)

        #función de compatibilidad
        d = dy.tanh(dy.cmult(W3 * out_tree, W2 * matrix_tag))
        scores = dy.transpose(d) * W1

        # Función distribución.
        weights = dy.softmax(scores)

        #Weighted Sum
        context = matrix_tag * weights
        return context, weights

def get_attention(attention, di, dh=None, da=None, pc=None):
    if attention == 'att_tag':
        return Attention_Tag(di, dh, da, pc)


def plot_attention(attention, sentence, attention2, sentence2, predicted, label):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if predicted==0:
        etiqueta= 'Duplicado'
    else:
        etiqueta='No Duplicado'
    if label == 0:
        etiqueta_real = 'Duplicado'
    else:
        etiqueta_real = 'No Duplicado'
    fig.suptitle('Predicción: ' +str(etiqueta) + ' Label: '+str(etiqueta_real))
    attention = np.expand_dims(array(attention.value()), axis=1)
    attention = np.repeat(attention[:, np.newaxis],1, axis=1)
    cax = ax1.matshow(attention[:,:,0], interpolation='nearest')
    fontdict = {'fontsize': 8}
    ax1.set_yticklabels([''] + sentence, fontdict=fontdict)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    attention2 = np.expand_dims(array(attention2.value()), axis=1)
    attention2 = np.repeat(attention2[:, np.newaxis],1, axis=1)
    cax2 = ax2.matshow(attention2[:,:,0], interpolation='nearest')
    fontdict = {'fontsize': 8}
    ax2.set_yticklabels([''] + sentence2, fontdict=fontdict)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.colorbar(cax, ax=ax1)
    plt.colorbar(cax2, ax=ax2)
    plt.show()

def plot_attention_score(attention, sentence,score, attention2, sentence2,score2,  predicted, label):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if predicted==0:
        etiqueta= 'Duplicado'
    else:
        etiqueta='No Duplicado'
    if label == 0:
        etiqueta_real = 'Duplicado'
    else:
        etiqueta_real = 'No Duplicado'
    fig.suptitle('Predicción: ' +str(etiqueta) + ' Label: '+str(etiqueta_real))
    attention = np.expand_dims(array(attention.value()), axis=1)
    attention = np.repeat(attention[:, np.newaxis],1, axis=1)
    cax = ax1.matshow(attention[:,:,0], interpolation='nearest')
    fontdict = {'fontsize': 8}
    sentence_score = [i + '( '+str("%.3f" % j)+' )' for i, j in zip(sentence, score)]
    ax1.set_yticklabels([''] +sentence_score  , fontdict=fontdict)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    attention2 = np.expand_dims(array(attention2.value()), axis=1)
    attention2 = np.repeat(attention2[:, np.newaxis],1, axis=1)
    cax2 = ax2.matshow(attention2[:,:,0], interpolation='nearest')
    fontdict = {'fontsize': 8}
    sentence_score2 = [i + '( ' + str("%.3f" % j) + ' )' for i, j in zip(sentence2, score2)]
    ax2.set_yticklabels([''] + sentence_score2, fontdict=fontdict)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.colorbar(cax, ax=ax1)
    plt.colorbar(cax2, ax=ax2)
    plt.show()