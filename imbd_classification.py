import re

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
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

    data['token'] = tokenizer.texts_to_sequences(data['review'])
    print(data.head())

    embedding = tf.keras.layers.Embedding(99426, 5)
    res = embedding(data['token'][:1])
