import tensorflow as tf
from matplotlib import pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


if __name__ == '__main__':
    fashion = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(28, 5, activation='relu', input_shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Softmax()
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(train_images, train_labels, batch_size=32, epochs=5)
    model.evaluate(test_images, test_labels, verbose=1)
