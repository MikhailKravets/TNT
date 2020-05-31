import tensorflow as tf


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    x, y = x_test[:1], y_test[:1]
    predictions = model(x).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    init_loss = loss_fn(y, predictions).numpy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)
