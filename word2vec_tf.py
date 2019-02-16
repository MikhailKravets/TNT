"""Learning the word2vec model by implementing
tensorflow example
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

WINDOW_SIZE = 2
EMBEDDING_DIM = 5
N = 10_000

corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
corpus_raw = corpus_raw.lower()


def to_one_hot(data_point_index, size):
    temp = np.zeros(size)
    temp[data_point_index] = 1
    return temp


if __name__ == '__main__':
    words = set()
    for word in corpus_raw.split():
        if word != '.':
            words.add(word)

    word2int = {}
    int2word = {}

    vocab_size = len(words)

    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word

    sentences = [v.split() for v in corpus_raw.split('.')]

    data = []

    for s in sentences:
        for i, word in enumerate(s):
            for nb_word in s[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, len(s)) + 1]:
                if nb_word != word:
                    data.append([word, nb_word])

    x_train, y_train = [], []

    for x, y in data:
        x_train.append(to_one_hot(word2int[x], vocab_size))
        y_train.append(to_one_hot(word2int[y], vocab_size))

    x = tf.placeholder(tf.float32, shape=(None, vocab_size))
    y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

    W1 = tf.Variable(tf.random_normal(shape=(vocab_size, EMBEDDING_DIM)))
    b1 = tf.Variable(tf.random_normal(shape=(EMBEDDING_DIM,)))

    hidden = tf.add(tf.matmul(x, W1), b1)

    W2 = tf.Variable(tf.random_normal(shape=(EMBEDDING_DIM, vocab_size)))
    b2 = tf.Variable(tf.random_normal(shape=(vocab_size,)))

    output = tf.nn.softmax(tf.add(tf.matmul(hidden, W2), b2))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(output), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

    for i in tqdm(range(N)):
        sess.run(train_step, feed_dict={x: x_train, y_label: y_train})

    vectors = sess.run(W1 + b1)
