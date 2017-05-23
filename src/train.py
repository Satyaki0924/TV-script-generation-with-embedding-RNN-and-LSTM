import os
import pickle

import numpy as np
import tensorflow as tf

from src.graph import Graph
from src.preprocess import PreProcess


class Train(object):
    def __init__(self, num_epochs, batch_size, rnn_size,
                 embed_dim, seq_length, learning_rate, show_every_n_batches):
        self.file = os.path.dirname(os.path.realpath(__file__))
        self.int_text, self.int_to_vocab = PreProcess(os.path.dirname(
            os.path.realpath(self.file)) + '/data/simpsons/moes_tavern_lines.txt').preprocess_and_save_data()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.show_every_n_batches = show_every_n_batches
        self.save_dir = os.path.dirname(os.path.realpath(self.file))

    def save_params(self, params):
        pickle.dump(params, open(os.path.dirname(os.path.realpath(self.file)) + '/dump/params.p', 'wb'))

    def get_batches(self):
        int_text = np.array(self.int_text)
        num_batches = len(int_text) // (self.batch_size * self.seq_length)
        input = int_text[:num_batches * self.batch_size * self.seq_length]
        target = int_text[1:(num_batches * self.batch_size * self.seq_length) + 1]
        batches = np.zeros([num_batches, 2, self.batch_size, self.seq_length])

        x = np.array(np.split(input, indices_or_sections=self.batch_size))
        y = np.array(np.split(target, indices_or_sections=self.batch_size))
        for i in range(num_batches):
            batches[i][0] = x[:, i * self.seq_length: (i + 1) * self.seq_length]
            batches[i][1] = y[:, i * self.seq_length: (i + 1) * self.seq_length]
        return batches

    def train(self):
        batches = self.get_batches()
        train_graph, initial_state, input_text, targets, lr, cost, final_state, train_op = Graph(self.int_to_vocab,
                                                                                                 self.rnn_size,
                                                                                                 self.embed_dim).graph()
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(self.num_epochs):
                state = sess.run(initial_state, {input_text: batches[0][0]})
                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        input_text: x,
                        targets: y,
                        initial_state: state,
                        lr: self.learning_rate}
                    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
                    if (epoch_i * len(batches) + batch_i) % self.show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))
                    with open(os.path.dirname(os.path.realpath(self.file)) + '/dump/loss.txt', 'a') as f:
                        f.write(str(train_loss) + '\n')
                        f.close()
            saver = tf.train.Saver()
            saver.save(sess, self.save_dir)
            self.save_params((self.seq_length, self.save_dir))
            print('Model Trained and Saved')
