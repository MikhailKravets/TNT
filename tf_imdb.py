import tensorflow as tf

from core import load_imdb2pandas

if __name__ == '__main__':
    training, test = load_imdb2pandas(user="root",
                                      password="password",
                                      db="imdb",
                                      collection="reviews")
    training_set = tf.data.Dataset.from_tensor_slices({'x': training['text'].values, 'y': training['sentiment'].values})
    training_set = training_set.batch(32)

    iterator = training_set.make_one_shot_iterator()

    with tf.Session() as sess:
        val = sess.run(iterator.get_next())

