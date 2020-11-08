import re
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

NUM_EXAMPLES = 20_000  # Place -1 to load all examples
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024


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


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        output, state = self.gru(x)
        return output, state


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, hidden, training=None, mask=None):
        x = self.embedding(inputs)
        x, state = self.gru(x, initial_state=hidden)
        x = self.dense(x)
        return x, state


class Translator(tf.keras.Model):

    def __init__(self, encoder_vocab_size, decoder_vocab_size, embedding_dim, units):
        super().__init__()
        self.encoder = Encoder(encoder_vocab_size, embedding_dim, units)
        self.decoder = Decoder(decoder_vocab_size, embedding_dim, units)

    def call(self, inputs, training=None, mask=None):
        x, state = self.encoder(inputs[0])
        dec, _ = self.decoder(inputs[1], state)
        return dec, state

    def train_step(self, data):
        x, y = data
        print(x.numpy())


if __name__ == '__main__':
    path_to_file = BASE_DIR.joinpath(".data/deu.txt")
    inp_data, target_data = create_dataset(path_to_file, NUM_EXAMPLES)

    inp_tensor, inp_tokenizer = tokenize(inp_data)
    target_tensor, target_tokenizer = tokenize(target_data)

    input_train, input_val, target_train, target_val = train_test_split(inp_tensor, target_tensor, test_size=0.2)

    vocab_inp_size = len(inp_tokenizer.word_index) + 1
    vocab_tar_size = len(target_tokenizer.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(len(input_train))
    dataset = dataset.batch(BATCH_SIZE)

    sample_batch = next(iter(dataset.take(1)))
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS)
    sample_encoder_output, sample_state = encoder(sample_batch[0])

    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS)
    sample_decoder_output, sample_decoder_state = decoder(sample_batch[1], sample_state)

    print(f"Encoder output shape: {sample_encoder_output.shape}")
    print(f"State shape: {sample_state.shape}")
    print(f"Sample decoder shape: {sample_decoder_output.shape}")
    print(f"State shape (from dec. should be same): {sample_decoder_state.shape}")

    translator = Translator(
        encoder_vocab_size=vocab_inp_size,
        decoder_vocab_size=vocab_tar_size,
        embedding_dim=EMBEDDING_DIM,
        units=UNITS,
    )
    translator.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    translator.fit(dataset)  # TODO: make training as normal function, not class
