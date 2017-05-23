import tensorflow as tf


class LSTM(object):
    @staticmethod
    def get_inputs():
        inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
        return inputs, targets, learning_rate

    @staticmethod
    def get_init_cell(batch_size, rnn_size):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.7)
        cell = tf.contrib.rnn.MultiRNNCell([drop])
        initial_state = cell.zero_state(batch_size, tf.float32)
        initial_state = tf.identity(initial_state, name='initial_state')
        return cell, initial_state

    @staticmethod
    def get_embed(input_data, vocab_size, embed_dim):
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_data)
        return embed

    @staticmethod
    def build_rnn(cell, inputs):
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype='float32')
        final_state = tf.identity(final_state, name='final_state')
        return outputs, final_state

    @staticmethod
    def build_nn(cell, input_data, vocab_size, embed_dim):
        embed = LSTM().get_embed(input_data, vocab_size, embed_dim)
        outputs, final_state = LSTM().build_rnn(cell, embed)
        logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                   biases_initializer=tf.zeros_initializer())
        return logits, final_state
