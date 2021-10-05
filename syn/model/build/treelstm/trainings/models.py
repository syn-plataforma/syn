from collections import namedtuple

import dynet as dy
import numpy as np

StateTuple = namedtuple("StateTuple", ("h", "c"))


# Classifier for natural language inference (Bowman et al., ACL 2016)
class Classifier:
    def __init__(self, model, dim):
        self.b_nli = model.add_parameters((2 * dim,))  # NLI layer bias
        self.W_nli_1 = model.add_parameters((2 * dim, dim))  # NLI layer weight for first sentence
        self.W_nli_2 = model.add_parameters((2 * dim, dim))  # NLI layer weight for second sentence
        self.W_nli_u = model.add_parameters((2 * dim, dim))  # NLI layer weight for squared distance
        self.W_nli_v = model.add_parameters((2 * dim, dim))  # NLI layer weight for componentwise product
        self.b_s = model.add_parameters((2,))  # softmax bias
        self.w_s = model.add_parameters((2, 2 * dim))  # softmax weight

    # Returns the energy to be passed to a softmax
    def __call__(self, s1, s2):
        b_nli = dy.parameter(self.b_nli)
        W_nli_1 = dy.parameter(self.W_nli_1)
        W_nli_2 = dy.parameter(self.W_nli_2)
        W_nli_u = dy.parameter(self.W_nli_u)
        W_nli_v = dy.parameter(self.W_nli_v)
        u = dy.square(s1 - s2)
        v = dy.cmult(s1, s2)
        relu = dy.rectify(dy.affine_transform([b_nli, W_nli_1, s1, W_nli_2, s2, W_nli_u, u, W_nli_v, v]))

        b_s = dy.parameter(self.b_s)
        w_s = dy.parameter(self.w_s)
        return dy.affine_transform([b_s, w_s, relu])

class ClassifierCategorical:
    def __init__(self, model, dim, num_label):
        self.b_nli = model.add_parameters((2 * dim,))  # NLI layer bias
        self.W_nli_1 = model.add_parameters((2 * dim, dim))  # NLI layer weight for first sentence
        self.b_s = model.add_parameters((num_label,))  # softmax bias
        self.w_s = model.add_parameters((num_label, 2 * dim))  # softmax weight

    # Returns the energy to be passed to a softmax
    def __call__(self, s1):
        b_nli = dy.parameter(self.b_nli)
        W_nli_1 = dy.parameter(self.W_nli_1)
        relu = dy.rectify(dy.affine_transform([b_nli, W_nli_1, s1, ]))#W_nli_2, s2, W_nli_u, u, W_nli_v, v]))
        b_s = dy.parameter(self.b_s)
        w_s = dy.parameter(self.w_s)
        return dy.affine_transform([b_s, w_s, relu])



class Tree_Lstm:
    # Initialises the network's parameters and data structures

    def __init__(
            self,
            model,
            input_embeddings,
            update_embeddings=False,
            hidden_dim=512,
            order=0,
    ):
        self.order = int(order)
        ## Embeddings ##
        print('embedding_shape ', input_embeddings.shape[1])
        self.input_dim = input_embeddings.shape[1]
        self.hidden_dim = int(hidden_dim)
        self.dim_ds = int(hidden_dim)
        self.update_embeddings = bool(update_embeddings)
        self.embeddings = model.add_lookup_parameters(input_embeddings.shape)
        self.embeddings.init_from_array(input_embeddings)

        ## Weighting/gating mechanism ##

        self.w_score = model.add_parameters((self.hidden_dim,))
        self.inv_temp = 10
        self.num_states = 0
        self.len_in = 0
        # self.states_hidden = list()
        ## Tree-LSTM ##

        self.W = model.add_parameters((5 * self.hidden_dim, self.input_dim))
        self.U = model.add_parameters((5 * self.hidden_dim, 2 * self.hidden_dim))
        self.b = model.add_parameters((5 * self.hidden_dim,))

        print(
            "Set up CYK-Attention_out model with:\n" +
            "  order: " + str(self.order) + "\n" +
            "  input_dim: " + str(self.input_dim) + "\n" +
            "  hidden_dim: " + str(self.hidden_dim) + "\n" +
            "  update_embeddings: " + str(self.update_embeddings)
        )

    def tree_lstm(self, L=None, R=None, x=None):
        hid = self.hidden_dim
        W, U, b = dy.parameter(self.W), dy.parameter(self.U), dy.parameter(self.b)
        if L is not None and R is not None:
            preact = b + U * dy.concatenate([L.h, R.h])
            i = dy.logistic(preact[:hid])
            fL = dy.logistic(preact[hid:2 * hid] + 1.0)
            fR = dy.logistic(preact[2 * hid:3 * hid] + 1.0)
            o = dy.logistic(preact[3 * hid:4 * hid])
            u = dy.tanh(preact[4 * hid:])
            c = dy.cmult(fL, L.c) + dy.cmult(fR, R.c) + dy.cmult(i, u)
        else:
            preact = b + W * x
            i = dy.logistic(preact[:hid])
            o = dy.logistic(preact[3 * hid:4 * hid])
            u = dy.tanh(preact[4 * hid:])
            c = dy.cmult(i, u)
        h = dy.cmult(o, dy.tanh(c))
        return StateTuple(h=h, c=c)

    # Runs the network on a single parse tree, returning the sentence embedding
    def do_parse_tree(self, tree, len_in=None):
        # print(tree)
        self.len_in = len_in
        return self._do_parse_tree(tree)

    def _do_parse_tree(self, tree):
        if isinstance(tree, int):
            return self.tree_lstm(
                L=None,
                R=None,
                x=dy.lookup(self.embeddings, tree, update=self.update_embeddings),
            )

        elif isinstance(tree, tuple) and len(tree) == 2:
            return self.tree_lstm(
                L=self._do_parse_tree(tree[0]),
                R=self._do_parse_tree(tree[1]),
            )
        else:
            raise ValueError("Malformed tree: " + str(tree))
