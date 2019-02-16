"""Not working :("""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

WINDOW_SIZE = 2
EMBEDDING_DIM = 5
N = 1000

corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
corpus_raw = corpus_raw.lower()


def to_one_hot(data_point_index, size):
    temp = np.zeros(size)
    temp[data_point_index] = 1
    return temp


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_raw.split('.'))
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(corpus_raw.split('.'))
    seq_padded = pad_sequences(sequences, padding='post')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, input_shape=(5,))
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_crossentropy],
    )
    model.fit(seq_padded, seq_padded, epochs=100)
