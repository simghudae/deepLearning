import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argparser = argparse.ArgumentParser()
argparser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train', help="Directory where to write event logs and checkpoint.")
argparser.add_argument('--max_steps', type=int, default=1000000, help="""Number of batches to run.""")
argparser.add_argument('--log_device_placement', action='store_true', help="Whether to log device placement.")
argparser.add_argument('--log_frequency', type=int, default=10, help="How often to log results to the console.")

class VGG16(keras.models.Model):
    def __init__(self, input_shape):
        super(VGG16, self).__init__()

        weightDecay = 0.000
        self.numClasses = 10

        model = keras.models.Sequential()

        # 64
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # 128
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # 256
        model.add(keras.layers.Conv2D(256, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # 512
        model.add(keras.layers.Conv2D(512, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # 512
        model.add(keras.layers.Conv2D(512, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.5))

        # FC
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(weightDecay)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.numClasses))

        self.model = model

    def call(self, x):
        return self.model(x)


def normalize(xTrain, xTest):
    xTrain, xTest = xTrain / 255., xTest / 255.
    mean = np.mean(xTrain, axis=(0, 1, 2, 3))
    std = np.std(xTrain, axis=(0, 1, 2, 3))
    return (xTrain - mean) / (std + 1e-10), (xTest - mean) / (std + 1e-10)


def preparCifar(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.int32)


def computeLoss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def main():
    maxEpoch = 500
    tf.random.set_seed(22)

    (xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()
    xTrain, xTest = normalize(xTrain, xTest)

    trainLoader = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
    trainLoader = trainLoader.map(preparCifar).shuffle(50000).batch(256)

    testLoader = tf.data.Dataset.from_tensor_slices((xTest, yTest))
    testLoader = testLoader.map(preparCifar).shuffle(10000).batch(256)

    model = VGG16([32, 32, 3])

    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    metricTrain = keras.metrics.CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    for epoch in range(maxEpoch):

        for step, (x, y) in enumerate(trainLoader):
            y = tf.squeeze(y, axis=1)
            y = tf.one_hot(y, depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
                metricTrain.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                metricTest = keras.metrics.CategoricalAccuracy()
                for x, y in testLoader:
                    y = tf.squeeze(y, axis=1)
                    y = tf.one_hot(y, depth=10)
                    logits = model.predict(x)
                    metricTest.update_state(y, logits)

                print("epoch : {0}, step : {1}, loss : {2:.3f}, accTrain : {3:.3f}, accTest : {4:.3f}".format(epoch, step, float(loss), metricTrain.result().numpy(), metricTest.result().numpy()))
                metricTrain.reset_states()
                metricTest.reset_states()


if __name__ == '__main__':
    main()
