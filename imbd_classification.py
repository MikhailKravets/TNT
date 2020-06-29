import re
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


log_dir = ".data/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def preprocess_text(sen):
    sen = re.sub(r'[^a-zA-Z]', ' ', sen)
    sen = re.sub(r'\s+', ' ', sen)
    sen = re.sub(r'\s+[a-zA-Z]\s+', ' ', sen)
    return sen.lower()


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

    train_x, train_y = inp[2000:], np.expand_dims(data['sentiment'].values[2000:], 1)
    test_x, test_y = inp[:2000], np.expand_dims(data['sentiment'].values[:2000], 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(99427, 64, input_length=1600),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print(model.summary())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
    model.fit(train_x, train_y, batch_size=32,
              validation_data=(test_x, test_y),
              validation_steps=30, epochs=4,
              shuffle=True,
              callbacks=[tensorboard_callback])
