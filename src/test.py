import os
import pickle

import numpy as np
import tensorflow as tf


class Test(object):
    def __init__(self):
        self.file = os.path.dirname(os.path.realpath(__file__))

    def load_preprocess(self):
        return pickle.load(open(os.path.dirname(os.path.realpath(self.file)) + '/dump/preprocess.p', mode='rb'))

    def load_params(self):
        return pickle.load(open(os.path.dirname(os.path.realpath(self.file)) + '/dump/params.p', mode='rb'))

    @staticmethod
    def get_tensors(loaded_graph):
        return loaded_graph.get_tensor_by_name('input:0'), loaded_graph.get_tensor_by_name('initial_state:0'), \
               loaded_graph.get_tensor_by_name('final_state:0'), loaded_graph.get_tensor_by_name('probs:0')

    @staticmethod
    def pick_word(probabilities, int_to_vocab):
        choice = np.random.choice(len(int_to_vocab.keys()), p=probabilities)
        return int_to_vocab[choice]

    def test(self):
        gen_length = 200
        _, vocab_to_int, int_to_vocab, token_dict = Test().load_preprocess()
        seq_length, load_dir = Test().load_params()
        while True:
            try:
                prime_word = str(input('Enter a name (e.g: homer_simpson, moe_szyslak, or Barney_Gumble, etc.)\n>> '))
                break
            except Exception as e:
                print('*** Exception: ' + str(e) + ' ***')
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
            input_text, initial_state, final_state, probs = Test().get_tensors(loaded_graph)
            gen_sentences = [prime_word + ':']
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
            for n in range(gen_length):
                dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
                dyn_seq_length = len(dyn_input[0])
                probabilities, prev_state = sess.run(
                    [probs, final_state],
                    {input_text: dyn_input, initial_state: prev_state})
                pred_word = Test().pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)
                gen_sentences.append(pred_word)
            tv_script = ' '.join(gen_sentences)
            for key, token in token_dict.items():
                ending = ' ' if key in ['\n', '(', '"'] else ''
                tv_script = tv_script.replace(' ' + token.lower(), key)
            tv_script = tv_script.replace('\n ', '\n')
            tv_script = tv_script.replace('( ', '(')
            print(tv_script)
