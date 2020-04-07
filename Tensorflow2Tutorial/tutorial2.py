import os
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def prepareMnist(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def mnistDataset():
    (x, y), (xVal, yVal) = keras.datasets.fashion_mnist.load_data()
    y, yVal = tf.one_hot(y, depth=10), tf.one_hot(yVal, depth=10)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepareMnist)
    ds = ds.shuffle(60000).batch(100)

    dsVal = tf.data.Dataset.from_tensor_slices((xVal, yVal))
    dsVal = dsVal.map(prepareMnist)
    dsVal = dsVal.shuffle(10000).batch(100)
    return ds, dsVal


def main():
    trainDataset, valDataset = mnistDataset()

    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(trainDataset.repeat(), epochs=30, steps_per_epoch=500, validation_data=valDataset.repeat(), validation_steps=2)


if __name__ == '__main__':
    main()
