import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.python.data import Dataset
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

data_path = '.data/fra_cleared.csv'


def max_sequence_length(sequences):
    m = len(sequences[0])
    for v in sequences[1:]:
        if len(v) > m:
            m = len(v)
    return m


if __name__ == '__main__':
    dataset = pd.read_csv(data_path)[:20000]
    dataset['Target'] = dataset['French'].apply(lambda e: f"{e} _END")
    dataset['French'] = dataset['French'].apply(lambda e: f"START_ {e} _END")

    en_tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>@[\\]^`{|}~\t\n')
    en_tokenizer.fit_on_texts(dataset['English'])

    fr_tokenizer = Tokenizer(filters='!?"#$%&()*+,-./:;<=>@[\\]^`{|}~\t\n')
    fr_tokenizer.fit_on_texts(dataset['French'])

    en_inp_data = en_tokenizer.texts_to_sequences(dataset['English'])
    fr_inp_data = fr_tokenizer.texts_to_sequences(dataset['French'])
    fr_target_data = fr_tokenizer.texts_to_sequences(dataset['Target'])

    max_en, max_fr = max_sequence_length(en_inp_data), max_sequence_length(fr_inp_data)
    num_target_index = len(fr_tokenizer.word_index)

    en_inp_data = pad_sequences(en_inp_data, maxlen=max_en, padding='post')
    fr_inp_data = pad_sequences(fr_inp_data, maxlen=max_fr, padding='post')
    fr_target_data = pad_sequences(fr_target_data, maxlen=max_fr, padding='post')

    encoder_inp = layers.Input(shape=(None,))
    encoder_embedding = layers.Embedding(en_tokenizer.document_count, 64)(encoder_inp)
    encoder = layers.LSTM(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)

    decoder_inp = layers.Input(shape=(None,))
    decoder_embedding = layers.Embedding(fr_tokenizer.document_count, 64)
    dec_emb = decoder_embedding(decoder_inp)

    decoder = layers.LSTM(50, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder(dec_emb, initial_state=[state_h, state_c])

    decoder_dense = layers.Dense(num_target_index, activation='softmax')
    dense_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inp, decoder_inp], dense_outputs)
    model.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=0.0001, momentum=0.05),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['sparse_categorical_accuracy'],
    )
    print(model.summary())

    model.fit(
        [en_inp_data, fr_inp_data], fr_target_data,
        batch_size=128,
        epochs=15,
        validation_split=0.1
    )

    # Representation
    encoder_model = Model(encoder_inp, [state_h, state_c])

    decoder_state_input_h = layers.Input(shape=(50,))
    decoder_state_input_c = layers.Input(shape=(50,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding_predict = decoder_embedding(decoder_inp)
    decoder_outputs_predict, state_h_predict, state_c_predict = decoder(decoder_embedding_predict,
                                                                        initial_state=decoder_states_inputs)
    dense_outputs_predict = decoder_dense(decoder_outputs_predict)

    decoder_model = Model(
        [decoder_inp] + [decoder_state_input_h, decoder_state_input_c],
        [dense_outputs_predict] + [state_h_predict, state_c_predict]
    )

    def decode_texts(texts):
        sequences = pad_sequences(en_tokenizer.texts_to_sequences(texts), maxlen=max_en)
        states = encoder_model.predict(sequences)

        stop = False

        decode_start = pad_sequences(fr_tokenizer.texts_to_sequences(["START_"]), maxlen=max_fr)
        tokens = []
        current_len = 0

        while not stop and current_len < max_fr:
            output = decoder_model.predict([decode_start] + states)
            output = output[0]
            token = np.argmax(output[0, -1, 1:])

            if token == 2:
                stop = True
            else:
                tokens.append(token)

            current_len += 1
        return fr_tokenizer.sequences_to_texts([tokens])
