import re

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds


def preprocess_text(sen):
    sen = re.sub(r'[^a-zA-Z]', ' ', sen)
    sen = re.sub(r'\s+', ' ', sen)
    sen = re.sub(r'\s+[a-zA-Z]\s+', ' ', sen)
    return sen


if __name__ == '__main__':
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

    data = pd.read_csv('.data/IMDB.csv')
    data['sentiment'] = data['sentiment'].apply(lambda e: 1 if e == "positive" else 0)
    data['review'] = data['review'].apply(preprocess_text)
    print(data.head())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['review'])

    inp = pad_sequences(tokenizer.texts_to_sequences(data['review']))
    dataset = tf.data.Dataset.from_tensor_slices((inp, np.expand_dims(data['sentiment'].values, 1)))
    test_dataset = dataset.take(10_000)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(99426, 8),
        tf.keras.layers.GRU(8),
        tf.keras.layers.Dense(1, activation='relu'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(dataset, batch_size=8, validation_data=test_dataset, validation_steps=30)
