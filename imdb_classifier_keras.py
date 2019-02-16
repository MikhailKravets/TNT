import tensorflow as tf
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from matplotlib import pyplot as plot


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()

    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    test_data = pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
    train_data = pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

    vocab_size = 10_000

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16, input_shape=(None,)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    model.summary()

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(
        partial_x_train,
        partial_y_train,
        batch_size=32,
        epochs=40,
        validation_data=(x_val, y_val),
    )

    results = model.evaluate(test_data, test_labels)
    print(results)

    acc, loss, val_acc = history.history['acc'], history.history['loss'], history.history['val_acc']
    epochs = range(1, len(acc) + 1)

    plot.plot(epochs, acc, 'bo', 'Loss')
    plot.plot(epochs, val_acc, 'b', 'Validation')
    plot.show()
