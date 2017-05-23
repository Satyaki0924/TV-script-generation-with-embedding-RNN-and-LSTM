import tensorflow as tf
from tensorflow.contrib import seq2seq

from src.lstm import LSTM


class Graph(object):
    def __init__(self, int_to_vocab, rnn_size, embed_dim):
        self.int_to_vocab = int_to_vocab
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim

    def graph(self):
        train_graph = tf.Graph()
        with train_graph.as_default():
            vocab_size = len(self.int_to_vocab)
            input_text, targets, lr = LSTM().get_inputs()
            input_data_shape = tf.shape(input_text)
            cell, initial_state = LSTM().get_init_cell(input_data_shape[0], self.rnn_size)
            logits, final_state = LSTM().build_nn(cell, input_text, vocab_size, self.embed_dim)
            probs = tf.nn.softmax(logits, name='probs')
            cost = seq2seq.sequence_loss(
                logits,
                targets,
                tf.ones([input_data_shape[0], input_data_shape[1]]))
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            return train_graph, initial_state, input_text, targets, lr, cost, final_state, train_op