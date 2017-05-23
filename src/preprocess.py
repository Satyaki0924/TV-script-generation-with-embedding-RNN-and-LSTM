import os
import pickle
from collections import Counter


class PreProcess(object):
    def __init__(self, path):
        self.path = path
        self.file = os.path.dirname(os.path.realpath(__file__))
        self.text = None
        self.int_to_vocab = None
        self.vocab_to_int = None
        self.token_dict = None

    def create_lookup_tables(self):
        word_count = Counter(self.text)
        sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
        self.int_to_vocab = {count: word for count, word in enumerate(sorted_vocab)}
        self.vocab_to_int = {word: count for count, word in self.int_to_vocab.items()}

    def token_lookup(self):
        self.token_dict = {
            '.': '||Period||',
            ',': '||Comma||',
            '"': '||Quotation_Mark||',
            ';': '||Semicolon||',
            '!': '||Exclamation_mark||',
            '?': '||Question_mark||',
            '(': '||Left_Parentheses||',
            ')': '||Right_Parentheses||',
            '--': '||Dash||',
            '\n': '||Return||'
        }

    def load_data(self):
        input_file = os.path.join(self.path)
        with open(input_file, "r") as f:
            self.text = f.read()

    def preprocess_and_save_data(self):
        self.load_data()
        self.text = self.text[81:]
        self.token_lookup()
        for key, token in self.token_dict.items():
            self.text = self.text.replace(key, ' {} '.format(token))
        self.text = self.text.lower()
        self.text = self.text.split()
        self.create_lookup_tables()
        int_text = [self.vocab_to_int[word] for word in self.text]
        pickle.dump((int_text, self.vocab_to_int, self.int_to_vocab, self.token_dict),
                    open(os.path.dirname(os.path.realpath(self.file)) + '/dump/preprocess.p', 'wb'))
        return int_text, self.int_to_vocab
