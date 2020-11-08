import os
import re
from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent.parent


def preprocess_sentence(w):
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'

    return w


def create_dataset(path, num_examples):
    with open(path) as f:
        lines = f.read().strip().split('\n')
        word_pairs = [[preprocess_sentence(w) for w in line.split('\t')[:-1]] for line in lines[:num_examples]]

    return zip(*word_pairs)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


if __name__ == '__main__':
    path_to_file = BASE_DIR.joinpath(".data/deu.txt")
    inp_data, target_data = create_dataset(path_to_file, -1)

    inp_tensor, inp_tokenizer = tokenize(inp_data)
    target_tensor, target_tokenizer = tokenize(target_data)
