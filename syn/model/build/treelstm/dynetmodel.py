import math
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Union, Type

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger
from syn.model.build.treelstm.dynetconfig import get_dynet

dy = get_dynet()
load_environment_variables()
log = set_logger()


class Attention(object):
    def __init__(
            self,
            pc_param: dy.ParameterCollection = None,
            hidden_dim: int = 300,
            att_dim: int = 32,
            att_vector_dim: int = 1
    ):
        self.pc_param = pc_param
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.att_vector_dim = att_vector_dim

        self.W1 = self.pc_param.add_parameters((self.att_dim))
        self.W2 = self.pc_param.add_parameters((self.att_dim, self.att_vector_dim))
        self.W3 = self.pc_param.add_parameters((self.att_dim, self.hidden_dim))

        log.info(f"{self.__class__.__name__} parameters dimensions.")
        log.info(f"self.W1: {self.W1.shape()}")
        log.info(f"self.W2: {self.W2.shape()}")
        log.info(f"self.W3: {self.W3.shape()}")

    @abstractmethod
    def attend(self, leaf_state, leaf_hidden_state, attention_vector):
        """Attention"""
        raise NotImplementedError("This method must be inherited")

    def attention(self, leaf_state, leaf_hidden_state, attention_matrix):
        # Calculate Alignment Scores
        d = dy.tanh(dy.cmult(self.W3 * leaf_state, self.W2 * attention_matrix))
        scores = dy.transpose(d) * self.W1

        # Softmax alignment scores to obtain Attention weights
        weights = dy.softmax(scores)

        # Multiply Attention weights with attention vector to get context vector.
        context = attention_matrix * weights

        # Concatenating hidden state from Tree-LSTM with context vector.
        return dy.concatenate([leaf_hidden_state, context]), weights.npvalue()

    @staticmethod
    def _create_axe(ax, position, aspect, pad_fraction):
        width = axes_size.AxesY(ax, aspect=1. / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        divider = make_axes_locatable(ax)
        return divider.append_axes(position=position, size=width, pad=pad)

    @staticmethod
    def _create_colorbar_ticks(attention_weights):
        min_range = np.amin(attention_weights)
        max_range = np.amax(attention_weights)
        num_steps = attention_weights.size if attention_weights.size < 6 else 6
        step_range = (max_range - min_range) / num_steps
        if min_range < max_range:
            cbar_ticks = np.arange(start=min_range, stop=max_range + step_range, step=step_range)
        else:
            cbar_ticks = attention_weights

        return cbar_ticks

    def plot_attention(self, issue, attention_weights):
        cols = int(os.environ.get('PLOT_ATTENTION_COLS_GRID', 5))
        rows = math.ceil(len(attention_weights) / cols)
        fig = plt.figure(1)
        for i, sent in enumerate(issue):
            leaves = [leaf.label for leaf in sent.leaves_iter()]
            weights = attention_weights[i].reshape(-1, 1)

            ax = fig.add_subplot(rows, cols, i + 1)
            im = ax.imshow(weights)

            # Show all ticks
            ax.set_xticks([])
            ax.set_yticks(np.arange(len(leaves)))
            # Label ticks
            ax.set_yticklabels(leaves)

            # create an axes on the right side of ax.PLOT_ATTENTION_CBAR_POSITION
            cax = self._create_axe(
                ax,
                os.environ.get('PLOT_ATTENTION_CBAR_POSITION', 'right'),
                int(os.environ.get('PLOT_ATTENTION_CBAR_ASPECT', 40)),
                float(os.environ.get('PLOT_ATTENTION_CBAR_PAD_FRACTION', 0.5))
            )

            # Create colorbar
            cbar_ticks = self._create_colorbar_ticks(attention_weights[i])

            plt.colorbar(im, cax=cax, ticks=cbar_ticks)
            fig.tight_layout()

        plt.show()


class LeafAttention(Attention):
    def __init__(self, pc_param, hidden_dim, att_dim, att_vector_dim):
        super().__init__(pc_param, hidden_dim + att_vector_dim, att_dim, att_vector_dim)

    def attend(self, leaf_state, leaf_hidden_state, attention_vector):
        # Attention vector.
        attn_matrix = dy.scalarInput(attention_vector)

        return self.attention(leaf_state, leaf_hidden_state, attn_matrix)


class RootAttention(Attention):
    def __init__(self, pc_param, hidden_dim, att_dim, att_vector_dim):
        super().__init__(pc_param, hidden_dim, att_dim, att_vector_dim)

    def attend(self, leaf_state, leaf_hidden_state, attention_matrix):
        # Attention vector.
        attn_matrix = dy.transpose(dy.inputTensor(attention_matrix))

        return self.attention(leaf_state, leaf_hidden_state, attn_matrix)


def get_attention_vector_dim(method: str = 'leaf') -> int:
    attention_vector_dim: Dict[str, int] = {
        'none': 0,
        'leaf': 1,
        'root': 2
    }
    return attention_vector_dim[method]


def get_issue_description_lstm_input_dim(sentence_dim: int = 300, method: str = 'leaf') -> int:
    input_dim = (sentence_dim * 2) + get_attention_vector_dim(method)

    if 'root' != method:
        attn_vec_dim = get_attention_vector_dim(method)
        input_dim = ((sentence_dim + attn_vec_dim) * 2) + (1 * int(bool(attn_vec_dim)))

    return input_dim


class TreeLSTMBuilder(object):
    def __init__(self, pc_param, pc_embed, word_vocab, wdim, hdim, word_embed=None, attention: Attention = None):
        self.pc_param = pc_param
        self.pc_embed = pc_embed
        self.attention = attention

        # LeafAttention
        rows = hdim
        columns = hdim
        if isinstance(self.attention, LeafAttention):
            rows = hdim + self.attention.att_vector_dim
            columns = columns + self.attention.att_vector_dim * 2

        self.WS = [self.pc_param.add_parameters((rows, wdim)) for _ in "iou"]
        self.US = [self.pc_param.add_parameters((rows, 2 * columns)) for _ in "iou"]
        self.UFS = [self.pc_param.add_parameters((rows, 2 * columns)) for _ in "ff"]
        # self.UFS = [self.pc_param.add_parameters((rows, 2 * columns), name=f"Uf{suffix} (Tree-LSTM)") for suffix in
        #             ['1', '2']]
        self.BS = [self.pc_param.add_parameters(rows) for _ in "iouf"]
        self.E = self.pc_embed.add_lookup_parameters((len(word_vocab), wdim), init=word_embed)
        self.w2i = word_vocab

        log.info(f"Tree-LSTM builder parameters dimensions.")
        log.info(f"self.WS: {self.WS[0].shape()}")
        log.info(f"self.US: {self.US[0].shape()}")
        log.info(f"self.UFS: {self.UFS[0].shape()}")
        log.info(f"self.BS: {self.BS[0].shape()}")

        log.info(f"Word embeddings lookup parameters dimension.")
        log.info(f"self.E: {self.E.dim()}")
        log.info(f"Vocabulary dimension.")
        log.info(f"self.w2i: {len(self.w2i)}")

    def expr_for_tree(self, tree, decorate=False):

        if tree.is_leaf():
            raise RuntimeError('Tree structure error: meet with leaves')
        if len(tree.children) == 1:
            if not tree.children[0].is_leaf():
                raise RuntimeError('Tree structure error: tree nodes with one child should be a leaf')
            # Lookup word embeddings for leaf or get '_UNK_' word embeddings.
            emb = self.E[self.w2i.get(tree.children[0].label, 0)]
            Wi, Wo, Wu = [w for w in self.WS]
            bi, bo, bu, _ = [b for b in self.BS]

            # Calculation of the score for a neural network layer (e.g. b+Wz) where b is the bias, W is the weight
            # matrix, and z is the input. In this case xs[0] = b, xs[1] = W, and xs[2] = z.
            # dy.affine_transform(xs) = xs[0] + xs[1]*xs[2] + xs[3]*xs[4] + ...
            i = dy.logistic(dy.affine_transform([bi, Wi, emb]))
            o = dy.logistic(dy.affine_transform([bo, Wo, emb]))
            u = dy.tanh(dy.affine_transform([bu, Wu, emb]))

            # Multiply two expressions component-wise.
            c = dy.cmult(i, u)

            # Elementwise calculation of the hyperbolic tangent: dy.tanh(c).
            h = dy.cmult(o, dy.tanh(c))

            if decorate:
                tree._e = h

            # LeafAttention.
            if isinstance(self.attention, LeafAttention):
                h, _ = self.attention.attend(c, h, tree.label)

            # Cell output.
            return h, c
        if len(tree.children) != 2:
            raise RuntimeError('Tree structure error: only binary trees are supported.')
        e1, c1 = self.expr_for_tree(tree.children[0], decorate)

        e2, c2 = self.expr_for_tree(tree.children[1], decorate)

        # Concatenate columns
        e = dy.concatenate([e1, e2])

        Ui, Uo, Uu = [u for u in self.US]
        Uf1, Uf2 = [u for u in self.UFS]
        bi, bo, bu, bf = [b for b in self.BS]

        # Calculation of the score for a neural network layer (e.g. b+Wz) where b is the bias, W is the weight
        # matrix, and z is the input. In this case xs[0] = b, xs[1] = W, and xs[2] = z.
        # dy.affine_transform(xs) = xs[0] + xs[1]*xs[2] + xs[3]*xs[4] + ...
        i = dy.logistic(dy.affine_transform([bi, Ui, e]))
        o = dy.logistic(dy.affine_transform([bo, Uo, e]))
        f1 = dy.logistic(dy.affine_transform([bf, Uf1, e]))
        f2 = dy.logistic(dy.affine_transform([bf, Uf2, e]))
        u = dy.tanh(dy.affine_transform([bu, Uu, e]))

        # Multiply two expressions component-wise.
        c = dy.cmult(i, u) + dy.cmult(f1, c1) + dy.cmult(f2, c2)

        # Elementwise calculation of the hyperbolic tangent: dy.tanh(c).
        h = dy.cmult(o, dy.tanh(c))

        if decorate:
            tree._e = h

        # LeafAttention.
        if isinstance(self.attention, LeafAttention):
            h, _ = self.attention.attend(c, h, tree.label)

        return h, c

    def get_sentence_repr(self, issue_description, decorate=False):
        sent_repr = []
        attention_weights = []
        for i, sent in enumerate(issue_description[0]):
            h, c = self.expr_for_tree(
                tree=sent,
                decorate=decorate
            )

            # RootAttention.
            if isinstance(self.attention, RootAttention):
                h, attn_weights = self.attention.attend(c, h, issue_description[1][i])
                attention_weights.append(attn_weights)

            sent_repr.append(dy.concatenate([h, c]))

        return sent_repr, attention_weights

    # support saving:
    def param_collection(self):
        return self.pc_param, self.pc_embed

    @staticmethod
    def from_spec(spec, pc_param, pc_embed):
        word_vocab, wdim, hdim, word_embed, attention = spec
        return TreeLSTMBuilder(pc_param, pc_embed, word_vocab, wdim, hdim, word_embed, attention)


class IssueDescriptionBuilder(object):
    def __init__(self, pc_param, pc_embed, w2i, word_embed, params):
        self.pc_param = pc_param
        self.pc_embed = pc_embed
        self.params = params.copy()

        # Issue description builders (bi-LSTM).
        self.forward_issue_description_builder = dy.LSTMBuilder(
            layers=self.params['num_layers'],
            input_dim=get_issue_description_lstm_input_dim(self.params['sentence_hidden_dim'],
                                                           self.params['attention']),
            hidden_dim=self.params['sentence_hidden_dim'],
            model=self.pc_param
        )
        self.backward_issue_description_builder = dy.LSTMBuilder(
            layers=self.params['num_layers'],
            input_dim=get_issue_description_lstm_input_dim(self.params['sentence_hidden_dim'],
                                                           self.params['attention']),
            hidden_dim=self.params['sentence_hidden_dim'],
            model=self.pc_param
        )

        log.info(f"LSTM builders parameters dimensions.")
        for i, param in enumerate(self.forward_issue_description_builder.get_parameters()[0]):
            log.info(f"self.forward_issue_description_builder._{i}: {param.shape()}")

        for i, param in enumerate(self.backward_issue_description_builder.get_parameters()[0]):
            log.info(f"self.backward_issue_description_builder._{i}: {param.shape()}")

        # The Attention Mechanism is defined in a separate class
        self.attention = None
        if 'leaf' == self.params['attention']:
            self.attention = LeafAttention(
                self.pc_param,
                self.params['sentence_hidden_dim'],
                self.params['attention_dim'],
                get_attention_vector_dim(self.params['attention'])
            )
        elif 'root' == self.params['attention']:
            self.attention = RootAttention(
                self.pc_param,
                self.params['sentence_hidden_dim'],
                self.params['attention_dim'],
                get_attention_vector_dim(self.params['attention'])
            )

        # Sentence builder.
        self.sentence_builder = TreeLSTMBuilder(
            self.pc_param,
            self.pc_embed,
            w2i,
            self.params['embeddings_size'],
            self.params['sentence_hidden_dim'],
            word_embed,
            self.attention
        )

        self.spec = (w2i, word_embed, params)

    def get_issue_description_repr(self, issue_description):
        # Sentences.
        sent_expr, attention_weights = self.sentence_builder.get_sentence_repr(issue_description)
        num_sent = len(sent_expr)

        # Initialize vectors.
        fwd_i = [i for i in range(0, num_sent)]
        bwd_i = [i for i in range(0, num_sent)]

        # Initialize a new graph, and a new sequence.
        fwd_init = self.forward_issue_description_builder.initial_state()
        bwd_init = self.backward_issue_description_builder.initial_state()

        # Input Expressions from sentence representation.
        for b in range(0, num_sent):
            fwd_i[b] = sent_expr[b]
            bwd_i[b] = sent_expr[num_sent - 1 - b]

        # Hidden vectors.
        fwd_output = [x.output() for x in fwd_init.add_inputs(fwd_i)]
        bwd_output = [x.output() for x in bwd_init.add_inputs(bwd_i)]

        # Concatenate hidden vectors.
        fwd_bwd_output = []
        for fwd, bwd in zip(fwd_output, bwd_output):
            fwd_bwd_output.append(dy.concatenate([fwd, bwd]))

        # Averaging hidden vectors as issue description representation
        if len(fwd_bwd_output) == 0:
            print("empty")
        return dy.average(fwd_bwd_output), attention_weights

    # support saving:
    def param_collection(self):
        return self.pc_param, self.pc_embed

    @staticmethod
    def from_spec(spec, pc_param, pc_embed):
        w2i, word_embed, params = spec
        return IssueDescriptionBuilder(pc_param, pc_embed, w2i, word_embed, params)


class Layer(object):
    def __init__(self, input_dim, output_dim, dropout_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate


class IssueStructuredDataBuilder(object):
    def __init__(self, pc_param, params):
        self.pc_param = pc_param
        self.params = params.copy()
        self.dropout_active = True

        self.WSD = []
        self.layers = []

        # First hidden layer.
        first_layer = Layer(
            self.params['structured_data_input_dim'],
            self.params['structured_data_hidden_dim'],
            self.params['structured_data_dropout_rate']
        )

        # Register parameters in model
        self.append_layer(first_layer)

        # Rest of hidden layers.
        for i in range(self.params['structured_data_num_layers']):
            layer = Layer(
                self.params['structured_data_hidden_dim'],
                self.params['structured_data_hidden_dim'],
                self.params['structured_data_dropout_rate']
            )
            self.append_layer(layer)

        log.info(f"Structured data builder parameters dimensions.")
        for i in range(len(self.WSD)):
            log.info(f"self.WSD[{i}][0] (W): {self.WSD[i][0].shape()}")
            log.info(f"self.WSD[{i}][1] (b): {self.WSD[i][1].shape()}")

    def append_layer(self, layer: Layer):
        # Add to layers
        self.layers.append(layer)

        # Register parameters
        W = self.pc_param.add_parameters((layer.output_dim, layer.input_dim))
        b = self.pc_param.add_parameters((layer.output_dim))
        self.WSD.append((W, b))

    def get_issue_structured_data_repr(self, structured_data):
        # Expression for the current hidden state
        h_cur = dy.inputTensor(structured_data)

        for i in range(len(self.layers)):
            # Initialize parameters in computation.
            W = self.WSD[i][0]
            b = self.WSD[i][1]

            # Apply activation function
            h = dy.logistic(dy.affine_transform([b, W, h_cur]))

            # Take care of dropout
            if self.layers[i].dropout_rate > 0:
                if self.dropout_active:
                    # During training, drop random units
                    mask = dy.random_bernoulli((self.layers[i].output_dim), 1 - self.layers[i].dropout_rate)
                    h_dropped = dy.cmult(h, mask)
                else:
                    # At test time, multiply by the retention rate to scale
                    h_dropped = h * (1 - self.layers[i].dropout_rate)
            else:
                # If there's no dropout, don't do anything
                h_dropped = h

            # Set current hidden state
            h_cur = h_dropped

        return h_cur

    # support saving:
    def param_collection(self):
        return self.pc_param

    @staticmethod
    def from_spec(spec, pc_param):
        params = spec
        return IssueStructuredDataBuilder(pc_param, params)


class Classifier(object):
    def __init__(self, n_classes, w2i, word_embed, params, model_meta_file=None):
        self.params = params.copy()

        # Load input params.
        if model_meta_file is not None and model_meta_file != '':
            log.info(f"Loading hyperparameters from: '{str(Path(model_meta_file))}'.")
            saved_params = np.load(str(Path(model_meta_file)), allow_pickle=True).item()
            self.params.update(saved_params)

        # Add parameters for issue description representation.
        self.pc_param = dy.ParameterCollection()

        # Add parameters for word embeddings.
        self.pc_embed = dy.ParameterCollection()

    @abstractmethod
    def get_logits(self, issue_repr):
        raise NotImplementedError("This method must be inherited")

    @abstractmethod
    def classify(self, issue_description, structured_data):
        raise NotImplementedError("This method must be inherited")

    @abstractmethod
    def losses_batch(self, batch, loss_function):
        raise NotImplementedError("This method must be inherited")

    def objective(self, logits, label, loss_function):
        return self.__class__.__getattribute__(self, loss_function)(logits, label)

    @staticmethod
    def cross_entropy_loss(logits, label):
        return dy.pickneglogsoftmax(logits, label)

    @staticmethod
    def hinge_loss(logits, label):
        hl = dy.hinge(logits, label, 10.0)
        return hl

    def regularization_loss(self, coef):
        losses = [dy.l2_norm(p) ** 2 for p in self.pc_param.parameters_list()]
        return (coef / 2) * dy.esum(losses)

    def save(self, save_dir, model_name):
        meta_path = os.path.join(save_dir, 'meta', model_name)
        param_path = os.path.join(save_dir, 'param', model_name)
        embed_path = os.path.join(save_dir, 'embed', model_name)

        log.info(f"Saving params to: '{meta_path}'.")
        np.save(meta_path, self.params)
        log.info(f"Saving pc_param to: '{param_path}'.")
        tic = time.time()
        self.pc_param.save(param_path)
        log.info(f"Saving pc_param total time: {(time.time() - tic) / 60} minutes")

        log.info(f"Saving pc_embed to: '{embed_path}'.")
        tic = time.time()
        self.pc_embed.save(embed_path)
        log.info(f"Saving pc_embed total time: {(time.time() - tic) / 60} minutes")

        return meta_path + '.npy'

    def load_param_embed(self, model_meta_file):
        model_meta_file = model_meta_file.replace('.npy', '')
        param_path = model_meta_file.replace('meta', 'param')
        embed_path = model_meta_file.replace('meta', 'embed')
        log.info(f"Loading pc_param from: '{param_path}'.")
        tic = time.time()
        self.pc_param.populate(param_path)
        log.info(f"Loading pc_param total time: {(time.time() - tic) / 60} minutes")

        log.info(f"Loading pc_embed from: '{embed_path}'.")
        tic = time.time()
        self.pc_embed.populate(embed_path)
        log.info(f"Loading pc_embed total time: {(time.time() - tic) / 60} minutes")

    @staticmethod
    def delete(model_meta_file):
        if model_meta_file is None:
            return
        os.remove(model_meta_file)
        model_meta_file = model_meta_file.replace('.npy', '')
        os.remove(model_meta_file.replace('meta', 'param'))
        os.remove(model_meta_file.replace('meta', 'embed'))


class IssueClassifier(Classifier):
    def __init__(self, n_classes, w2i, word_embed, params, model_meta_file=None):
        super().__init__(n_classes, w2i, word_embed, params, model_meta_file)

        log.info(f"Classifier parameters dimensions.")

        # Issue description builder.
        log.info(f"Issue description builder parameters dimensions.")
        self.issue_description_builder = IssueDescriptionBuilder(
            self.pc_param,
            self.pc_embed,
            w2i,
            word_embed,
            self.params
        )

        # Structured data builder.
        if self.params['use_structured_data']:
            log.info(f"Issue structured data builder parameters dimensions.")
            self.structured_data_builder = IssueStructuredDataBuilder(
                self.pc_param,
                self.params
            )

        # Issue representation dimension
        issue_repr_dim = self.params['sentence_hidden_dim'] * 2
        if self.params['use_structured_data']:
            issue_repr_dim = issue_repr_dim + self.params['structured_data_hidden_dim']

        log.info(f"Fully connected layer dimensions.")
        # Layer: bias
        self.p_hbias = self.pc_param.add_parameters((issue_repr_dim,))
        log.info(f"self.p_hbias: {self.p_hbias.shape()}")

        # Layer: issue to hidden
        self.p_i2h = self.pc_param.add_parameters((issue_repr_dim, issue_repr_dim))
        log.info(f"self.p_hbias: {self.p_hbias.shape()}")

        # Layer: output bias
        self.p_obias = self.pc_param.add_parameters((n_classes,))
        log.info(f"self.p_obias: {self.p_obias.shape()}")

        # Layer: hidden to output
        self.p_h2o = self.pc_param.add_parameters((n_classes, issue_repr_dim))
        log.info(f"self.p_h2o: {self.p_h2o.shape()}")

        if model_meta_file is not None and model_meta_file != '':
            self.load_param_embed(model_meta_file)

    def get_logits(self, issue_repr):
        h = dy.rectify(dy.affine_transform([self.p_hbias, self.p_i2h, issue_repr]))
        logits = self.p_obias + (self.p_h2o * h)

        return logits

    def classify(self, issue_description, issue_structured_data):
        # Description representation.
        issue_description_repr, attention_weights = self.issue_description_builder.get_issue_description_repr(
            issue_description
        )

        elements_repr = [
            issue_description_repr
        ]
        if self.params['use_structured_data']:
            # Structured data representation.
            issue_structured_data_repr = self.structured_data_builder.get_issue_structured_data_repr(
                issue_structured_data)
            elements_repr.append(issue_structured_data_repr)

        # Issue representation.
        issue_repr = dy.concatenate(elements_repr)

        logits = self.get_logits(issue_repr)
        return logits, attention_weights

    def losses_batch(self, batch, loss_function):
        batch_losses = []
        for inst in batch:
            # Tuple(inst[0], inst[1], inst[2], inst[3]).
            # Tuple(trees, attention_vectors, structured_data, label).

            # Check data integrity.
            num_trees = len(inst[0])
            num_attn_vectors = len(inst[1])
            assert num_trees == num_attn_vectors, \
                f"Distinct length of trees '{num_trees}' and attention vectors '{num_attn_vectors}'."

            # Issue description as Tuple(trees, attention_vectors).
            issue_description = (inst[0], inst[1])

            # Issue structured data
            issue_structured_data = inst[2]

            logits, _ = self.classify(issue_description, issue_structured_data)
            label = inst[3]
            losses = self.objective(logits, label, loss_function)
            batch_losses.append(losses)
        return dy.esum(batch_losses)

    def predict(self, issue_description, issue_structured_data):
        logits, attention_weights = self.classify(issue_description, issue_structured_data)
        out = dy.softmax(logits)
        predict_proba = out.npvalue()
        pred = np.argmax(out.npvalue())

        return pred, predict_proba, attention_weights


class IssueSimilarityMeter(Classifier):
    def __init__(self, n_classes, w2i, word_embed, params, model_meta_file=None):
        super().__init__(n_classes, w2i, word_embed, params, model_meta_file)

        log.info(f"Classifier parameters dimensions.")

        # Left issue description builder.
        log.info(f"Issue description builder left parameters dimensions.")
        self.issue_description_builder_left = IssueDescriptionBuilder(
            self.pc_param,
            self.pc_embed,
            w2i,
            word_embed,
            self.params
        )

        # Left structured data builder.
        if self.params['use_structured_data']:
            log.info(f"Issue structured data builder left parameters dimensions.")
            self.structured_data_builder_left = IssueStructuredDataBuilder(
                self.pc_param,
                self.params
            )

        # Right issue description builder.
        log.info(f"Issue description builder right parameters dimensions.")
        self.issue_description_builder_right = IssueDescriptionBuilder(
            self.pc_param,
            self.pc_embed,
            w2i,
            word_embed,
            self.params
        )

        # Right structured data builder.
        if self.params['use_structured_data']:
            log.info(f"Issue structured data builder right parameters dimensions.")
            self.structured_data_builder_right = IssueStructuredDataBuilder(
                self.pc_param,
                self.params
            )

        # Issue representation dimension
        issue_repr_dim = self.params['sentence_hidden_dim'] * 2
        if self.params['use_structured_data']:
            issue_repr_dim = issue_repr_dim + self.params['structured_data_hidden_dim']

        log.info(f"Fully connected layer dimensions.")
        # Layer: bias
        self.p_hbias = self.pc_param.add_parameters((issue_repr_dim * 4,))
        log.info(f"self.p_hbias: {self.p_hbias.shape()}")

        # Layer: issue to hidden
        self.p_i2h = self.pc_param.add_parameters((issue_repr_dim * 4, issue_repr_dim * 4))
        log.info(f"self.p_i2h: {self.p_i2h.shape()}")

        # Layer: output bias
        self.p_obias = self.pc_param.add_parameters((n_classes,))
        log.info(f"self.p_obias: {self.p_obias.shape()}")

        # Layer: hidden to output
        self.p_h2o = self.pc_param.add_parameters((n_classes, issue_repr_dim * 4))
        log.info(f"self.p_h2o: {self.p_h2o.shape()}")

        if model_meta_file is not None and model_meta_file != '':
            self.load_param_embed(model_meta_file)

    def get_logits(self, issues_similarity_repr):
        h = dy.rectify(dy.affine_transform([self.p_hbias, self.p_i2h, issues_similarity_repr]))
        logits = self.p_obias + (self.p_h2o * h)

        return logits

    def classify(self, issue_description: list = None, issue_structured_data: list = None):
        # Issue descriptions representations.
        issue_description_left_repr, attention_weights_left = \
            self.issue_description_builder_left.get_issue_description_repr(issue_description[0])
        issue_description_right_repr, attention_weights_right = \
            self.issue_description_builder_right.get_issue_description_repr(issue_description[1])

        if self.params['use_structured_data']:
            # Left issue structured data representations.
            issue_structured_data_left_repr = self.structured_data_builder_left.get_issue_structured_data_repr(
                issue_structured_data[0]
            )
            issue_left_repr = dy.concatenate([issue_description_left_repr, issue_structured_data_left_repr])

            # Right issue structured data representations.
            issue_structured_data_right_repr = self.structured_data_builder_right.get_issue_structured_data_repr(
                issue_structured_data[1]
            )
            issue_right_repr = dy.concatenate([issue_description_right_repr, issue_structured_data_right_repr])
        else:
            issue_left_repr = issue_description_left_repr
            issue_right_repr = issue_description_right_repr

        # Left and right issues descriptions, and similarity representation.
        issues_similarity_repr = dy.concatenate(
            [
                dy.abs(issue_left_repr - issue_right_repr),
                dy.cmult(issue_left_repr, issue_right_repr),
                issue_left_repr,
                issue_right_repr
            ]
        )

        logits = self.get_logits(issues_similarity_repr)
        return logits, attention_weights_left, attention_weights_right

    def losses_batch(self, batch, loss_function):
        batch_losses = []
        for inst in batch:
            # Tuple(inst[0], inst[1], inst[2], inst[3], inst[4], inst[5], inst[6]) =
            # Tuple(trees_left, trees_right, attention_vectors_left, attention_vectors_right, structured_data_left,
            # structured_data_right, label).

            # Check data integrity.
            num_trees_left = len(inst[0])
            num_attn_vectors_left = len(inst[2])
            assert num_trees_left == num_attn_vectors_left, \
                f"Distinct length of trees_left '{num_trees_left}' and " \
                f"attention vectors_left '{num_attn_vectors_left}'."
            num_trees_right = len(inst[1])
            num_attn_vectors_right = len(inst[3])
            assert num_trees_right == num_attn_vectors_right, \
                f"Distinct length of trees_right '{num_trees_right}' and " \
                f"attention vectors_right '{num_attn_vectors_right}'."

            # Issue description as Tuple(trees, attention_vectors).
            issue_description_left = (inst[0], inst[2])
            issue_description_right = (inst[1], inst[3])

            # Issue structured data.
            issue_structured_data_left = inst[4]
            issue_structured_data_right = inst[5]

            # Compute logits
            logits, _, _ = self.classify(
                [issue_description_left, issue_description_right],
                [issue_structured_data_left, issue_structured_data_right]
            )

            # Compute loss
            label = inst[6]
            losses = self.objective(logits, label, loss_function)
            batch_losses.append(losses)
        return dy.esum(batch_losses)

    def predict(self, issue_description_left, issue_description_right,
                issue_structured_data_left, issue_structured_data_right):
        logits, attention_weights_left, attention_weights_right = self.classify(
            [issue_description_left, issue_description_right],
            [issue_structured_data_left, issue_structured_data_right]
        )
        out = dy.softmax(logits)
        predict_proba = out.npvalue()
        pred = np.argmax(predict_proba)

        return pred, predict_proba, attention_weights_left, attention_weights_right


def get_dynet_model(
        task: str = 'duplicity'
) -> Type[Union[IssueClassifier, IssueSimilarityMeter]]:
    dynet_model = {
        'assignation': IssueClassifier,
        'classification': IssueClassifier,
        'duplicity': IssueSimilarityMeter,
        'prioritization': IssueClassifier,
        'similarity': IssueSimilarityMeter
    }

    return dynet_model[task]
