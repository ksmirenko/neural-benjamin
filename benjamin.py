import os

import numpy as np
import re
from keras import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.models import load_model
from keras.optimizers import RMSprop

bb_songs_path = "data/lyrics/Breaking Benjamin"
bb_corpus_path = "data/bb_corpus.txt"
bb_model_path = "bb_model.h5"

song_separator = "#\n"
sequence_length = 50
sequence_step = 5

n_epochs = 30

generated_length = 500
n_options_for_char = 5


def fetch_all_text(path):
    corpus = ""
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            corpus += fetch_all_text(filepath)
            continue
        with open(filepath, mode='r', encoding='utf8') as f:
            corpus += f.read()
        corpus += song_separator
    return corpus


def prepare_corpus(corpus_path):
    corpus = fetch_all_text(bb_songs_path)
    corpus = corpus.lower()
    corpus = re.sub(r'[ ]+', ' ', corpus)
    corpus = re.sub(r'[^a-z0-9# \'\n]+', '', corpus)
    with open(corpus_path, mode='w+', encoding='utf8') as f:
        f.write(corpus)


def load_corpus(corpus_path):
    corpus = ""
    with open(corpus_path, mode='r', encoding='utf8') as f:
        corpus += f.read()
    chars = sorted(list(set(corpus)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    return corpus, chars, char_indices


def create_sequences(text, length, step):
    sequences = []
    next_chars = []
    for i in range(0, len(text) - length, step):
        sequences.append(text[i:(i + length)])
        next_chars.append(text[i + length])
    return sequences, next_chars


def train_model(corpus_path, model_path, n_epochs):
    corpus, chars, char_indices = load_corpus(corpus_path)
    sequences, next_chars = create_sequences(corpus, sequence_length, sequence_step)

    X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for sequence_index, sequence in enumerate(sequences):
        y[sequence_index, char_indices[next_chars[sequence_index]]] = 1
        for char_index, char in enumerate(sequence):
            X[sequence_index, char_index, char_indices[char]] = 1

    # Build the NN
    model = Sequential()
    model.add(LSTM(128, input_shape=(sequence_length, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Train
    model.fit(X, y, batch_size=128, epochs=n_epochs)
    model.save(model_path)


def generate_song(corpus_path, model_path, length, song_start):
    def gen_next_char(predictions):
        selected_probs = predictions
        selected_probs[np.argsort(selected_probs)[:-n_options_for_char]] = 0
        selected_probs /= selected_probs.sum()
        next_char = np.random.choice(chars, size=None, p=selected_probs)
        return next_char

    _, chars, char_indices = load_corpus(corpus_path)
    model = load_model(model_path)

    sentence = song_start
    generated = song_start
    for _ in range(length - len(song_start)):
        x = np.zeros((1, sequence_length, len(chars)))
        for char_index, char in enumerate(sentence):
            x[0, char_index, char_indices[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_char = gen_next_char(predictions)

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


prepare_corpus(bb_corpus_path)
train_model(bb_corpus_path, bb_model_path, n_epochs=n_epochs)
song = generate_song(corpus_path=bb_corpus_path,
                     model_path=bb_model_path,
                     length=generated_length,
                     song_start="border line\ndead inside\ni don't mind\nfalling to pi")
print(song)
