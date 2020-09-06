import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.python.data import Dataset
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

data_path = '.data/fra_cleared.csv'

max_en, max_fr = 44, 57  # Max length of sentence. Calculated previously

if __name__ == '__main__':
    dataset = pd.read_csv(data_path)[:5000]
    dataset['Target'] = dataset['French'].apply(lambda e: f"{e} <END>")
    dataset['French'] = dataset['French'].apply(lambda e: f"<START> {e} <END>")

    en_tokenizer = Tokenizer()
    en_tokenizer.fit_on_texts(dataset['English'])

    fr_tokenizer = Tokenizer()
    fr_tokenizer.fit_on_texts(dataset['French'])

    en_inp_data = pad_sequences(en_tokenizer.texts_to_sequences(dataset['English']), maxlen=max_en)
    fr_inp_data = pad_sequences(fr_tokenizer.texts_to_sequences(dataset['French']), maxlen=max_fr)

    # fr_target_data = pad_sequences(fr_tokenizer.texts_to_sequences(dataset['Target']), maxlen=max_fr)

    fr_target_data = np.zeros(shape=(fr_tokenizer.document_count, max_fr, len(fr_tokenizer.word_index)))

    encoder_inp = layers.Input(shape=(None,))
    encoder_embedding = layers.Embedding(en_tokenizer.document_count, 64)(encoder_inp)
    encoder = layers.LSTM(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)

    decoder_inp = layers.Input(shape=(None,))
    decoder_embedding = layers.Embedding(fr_tokenizer.document_count, 64)(decoder_inp)
    decoder = layers.LSTM(50, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=[state_h, state_c])

    decoder_dense = layers.Dense(len(fr_tokenizer.word_index), activation='softmax')
    dense_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inp, decoder_inp], dense_outputs)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    print(model.summary())
    print(fr_target_data.shape)

    # model.fit(
    #     [en_inp_data, fr_inp_data], fr_target_data,
    #     batch_size=12,
    #     epochs=50,
    # )
