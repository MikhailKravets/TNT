import re

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

    inp = pad_sequences(tokenizer.texts_to_sequences(data['review']), maxlen=1600)
    dataset = tf.data.Dataset.from_tensor_slices((inp, np.expand_dims(data['sentiment'].values, 1)))
    test_dataset = dataset.take(10_000)
    dataset = dataset.skip(10_000)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(99426, 64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(64, 3, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    print(model.summary())
    model.fit(dataset, batch_size=32, validation_data=test_dataset, validation_steps=30, epochs=4)
