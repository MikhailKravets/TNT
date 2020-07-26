import pandas
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.keras import layers, losses
from tensorflow.python.keras import Model


def convert_data():
    with open('.data/fra.txt', 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        inp, target, _ = line.split('\t')
        data.append((inp, target))

    data = pandas.DataFrame(data, columns=['English', 'French'])
    data.to_csv('.data/fra_cleared.csv', index=False)
    return data


if __name__ == '__main__':
    data = pandas.read_csv('.data/fra_cleared.csv')

    data['French'] = data['French'].apply(lambda e: f"<START> {e} <STOP>")

    filters = '"#$%&()*+-./:;<=>@[\\]^_`{|}~\t\n'
    eng_tokenizer, french_tokenizer = Tokenizer(filters=filters), Tokenizer(filters=filters)
    eng_tokenizer.fit_on_texts(data['English'])
    french_tokenizer.fit_on_texts(data['French'])

    encoder_words_count = len(eng_tokenizer.word_index)
    decoder_words_count = len(french_tokenizer.word_index)

    encoder_inp = layers.Input(shape=(None,))
    encoder_emb = layers.Embedding(encoder_words_count, 128)(encoder_inp)

    encoder = layers.LSTM(64, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_emb)

    decoder_inp = layers.Input(shape=(None,))
    decoder_emb = layers.Embedding(decoder_words_count, 128)(decoder_inp)

    decoder = layers.LSTM(64, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder(decoder_emb, initial_state=(state_h, state_c))

    dense = layers.Dense(decoder_words_count, activation='softmax')
    output = dense(decoder_output)

    encoder_input_vectors = pad_sequences(eng_tokenizer.texts_to_sequences(data['English']), maxlen=1600)
    decoder_input_vectors = pad_sequences(french_tokenizer.texts_to_sequences(data['French']), maxlen=1600)

    model = Model([encoder_inp, decoder_inp], output)
    model.compile(
        optimizer='rmsprop',
        loss=losses.CategoricalCrossentropy(),
        metrics=['acc']
    )
    print(model.summary())

    # TODO: connect tensorboard
    # TODO: connect checkpoints
    # TODO: how to create decoder_target_data??? Prepend <START> and <STOP> to sentences in French
