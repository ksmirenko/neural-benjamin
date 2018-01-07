import os
import re
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.models import load_model
from keras.optimizers import RMSprop

bb_songs_path = "data/lyrics/Breaking Benjamin"
bb_corpus_path = "data/bb_corpus.txt"
bb_new_model_path = "bb_model_112epochs.h5"

song_separator = "\n#\n"
sequence_length = 100
sequence_step = 1

n_epochs = 100

generated_length = 600
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


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'[^a-z0-9# \'\n]+', '', text)
    return text


def prepare_corpus(corpus_path):
    corpus = fetch_all_text(bb_songs_path)
    corpus = preprocess_text(corpus)
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


def train_model(corpus_path, new_model_path, n_epochs, old_model_path=None):
    corpus, chars, char_indices = load_corpus(corpus_path)
    sequences, next_chars = create_sequences(corpus, sequence_length, sequence_step)

    X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for sequence_index, sequence in enumerate(sequences):
        y[sequence_index, char_indices[next_chars[sequence_index]]] = 1
        for char_index, char in enumerate(sequence):
            X[sequence_index, char_index, char_indices[char]] = 1

    if old_model_path is None:
        # Build the model
        model = Sequential()
        model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
    else:
        # Optionally load the model instead of building a new one
        model = load_model(old_model_path)

    # Optional callbacks
    filepath = "data/intermediate/bb_model_improvement-{epoch:02d}-{loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Train
    model.fit(X, y, batch_size=128, epochs=n_epochs)
    model.save(new_model_path)


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


def prepare_song_start(text):
    return preprocess_text(text)[:sequence_length]


def write_song(out_path, song_start):
    song_start = prepare_song_start(song_start)
    if len(song_start) != sequence_length:
        print("Error: incorrect length of song start!")
        return
    song = generate_song(bb_corpus_path, bb_new_model_path, generated_length, song_start)
    with open(out_path, mode='w+', encoding='utf8') as f:
        f.write(song)


# prepare_corpus(bb_corpus_path)
# train_model(bb_corpus_path, new_model_path=bb_new_model_path, n_epochs=n_epochs)

write_song("out_songs/bb_3.txt", """"Shove me under you again
I can't wait for this to end
Sober, empty in the head
I know I can never win""")
