import numpy as np
import re
import pickle
import torch

class SimpleTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.total_words = 0

    def fit_on_texts(self, texts):
        words = set()
        for text in texts:
            words.update(text.split())
        
        self.word_index = {word: i + 1 for i, word in enumerate(sorted(list(words)))}
        self.index_word = {i + 1: word for i, word in enumerate(sorted(list(words)))}
        self.total_words = len(self.word_index) + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 0) for word in text.split()]
            sequences.append(seq)
        return sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_sequences(text, max_sequence_len=10):
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts([text])
    total_words = tokenizer.total_words
    
    # Tokenize the entire text once
    tokenized_text = tokenizer.texts_to_sequences([text])[0]
    
    input_sequences = []
    for i in range(1, len(tokenized_text)):
        n_gram_sequence = tokenized_text[max(0, i - max_sequence_len + 1): i + 1]
        input_sequences.append(n_gram_sequence)
        
    actual_max_len = max([len(x) for x in input_sequences])
    
    # Pad sequences manually
    padded_sequences = []
    for seq in input_sequences:
        pad_len = actual_max_len - len(seq)
        padded_seq = [0] * pad_len + seq
        padded_sequences.append(padded_seq)
    
    input_sequences = np.array(padded_sequences)
    
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    
    return X, y, total_words, actual_max_len, tokenizer

def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
