# Regerssion Code

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Regressor(keras.layers.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.W = self.add_variable('meanless-name', [13, 1])
        self.b = self.add_variable('meanless-name', [1])

    def call(self, x):
        x = tf.matmul(x, self.W) + self.b
        return x


def main():
    tf.random.set_seed(22)
    np.random.seed(22)

    (xTrain, yTrain), (xVal, yVal) = keras.datasets.boston_housing.load_data()
    xTrain, xVal = xTrain.astype(np.float32), xVal.astype(np.float32)

    dbTrain = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)).batch(64)
    dbVal = tf.data.Dataset.from_tensor_slices((xVal, yVal)).batch(102)

    model = Regressor()
    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    for epoch in range(10000):
        for step, (x, y) in enumerate(dbTrain):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criteon(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            for x, y in dbVal:
                testLogits = model(x)
                testLogits = tf.squeeze(testLogits, axis=1)
                testLoss = criteon(y, testLogits)
            print("epoch : {0}, trainLoss : {1:.3f}, testLoss : {2:.3f}".format(epoch, loss.numpy(), testLoss.numpy()))


if __name__ == '__main__':
    main()
