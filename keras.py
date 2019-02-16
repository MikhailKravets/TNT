import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(32,), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.01),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
    )

    model.fit(dataset, epochs=1000, steps_per_epoch=10, validation_data=val_dataset, validation_steps=3)
