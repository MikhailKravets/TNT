import re
import time
from pathlib import Path

import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

NUM_EXAMPLES = 100_000  # Place -1 to load all examples
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024
EPOCHS = 10
MAX_LEN_TARGET = 30


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


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# @tf.function
def train_step(inp, target, target_tokenizer, encoder, decoder, optimizer, metric):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, state = encoder(inp)

        dec_hidden = state

        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        pred = tf.expand_dims([target_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, target.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target[:, t], predictions, loss_object)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)
            pred = tf.concat([pred, tf.cast(tf.expand_dims(tf.argmax(predictions, axis=1), 1), dtype=tf.int32)], axis=1)

        metric.update_state(
            tf.expand_dims(target, 2),
            tf.expand_dims(pred, 2)
        )

    batch_loss = loss / target.shape[1]
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def accuracy(x, y, encoder, decoder, metric, target_tokenizer):
    enc_output, state = encoder(x)
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * len(y), 1)

    for i in range(y.shape[1]):
        predictions, state = decoder(dec_input, state, enc_output)
        max_id = tf.expand_dims(tf.argmax(predictions, axis=1), 1)

        metric.update_state(tf.expand_dims(y[:, i], 1), max_id)

        dec_input = max_id


def translate(sentence, inp_tokenizer, target_tokenizer, encoder, decoder):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_tokenizer.word_index[v] for v in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], padding='post')

    enc_output, state = encoder(inputs)

    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)

    result = ""
    for _ in range(MAX_LEN_TARGET):
        predictions, state = decoder(dec_input, state)

        max_id = tf.argmax(predictions[0]).numpy()
        result += target_tokenizer.index_word[max_id] + ' '

        if target_tokenizer.index_word[max_id] == '<end>':
            return result

        dec_input = tf.expand_dims([max_id], 0)

    return result


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


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
        self.attention = BahdanauAttention(self.units)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, hidden, enc_output, training=None, mask=None):
        context_vector = self.attention(hidden, enc_output)
        x = self.embedding(inputs)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.dense(output)
        return x, state


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

    sample_batch = next(iter(dataset.take(2)))
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS)

    input_x, input_y = sample_batch[0], tf.expand_dims(sample_batch[1][:, 0], 1)
    sample_encoder_output, sample_state = encoder(input_x)

    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS)
    sample_decoder_output, sample_decoder_state = decoder(input_y, sample_state, sample_encoder_output)

    print(f"Encoder output shape: {sample_encoder_output.shape}")
    print(f"State shape: {sample_state.shape}")
    print(f"Sample decoder shape: {sample_decoder_output.shape}")
    print(f"State shape (from dec. should be same): {sample_decoder_state.shape}")

    steps_per_epoch = len(input_train) // BATCH_SIZE
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    train_acc_metric = tf.keras.metrics.Accuracy()
    val_acc_metric = tf.keras.metrics.Accuracy()

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        batch_time = time.time()
        for i, (inp, target) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, target, target_tokenizer, encoder, decoder, optimizer, train_acc_metric)
            total_loss += batch_loss

            if i % 50 == 0:
                print(f"Epoch {epoch + 1} Batch {i} Loss {batch_loss:.4f}, Acc: {train_acc_metric.result()}")
                print(f"Time executing: {time.time() - batch_time} sec\n")
                batch_time = time.time()
                train_acc_metric.reset_states()

        val_acc_metric.reset_states()
        accuracy(input_val, target_val, encoder, decoder, val_acc_metric, target_tokenizer)

        # TODO: save checkpoints

        print(f"Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}. Val acc: {val_acc_metric.result()}")
        print(f"Time taken for 1 epoch {time.time() - start} sec\n")
