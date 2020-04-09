import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(xTrain, yTrain), (xTest, yTest) = keras.datasets.fashion_mnist.load_data()
xTrain, xTest = xTrain.astype(np.float32) / 255., xTest.astype(np.float32) / 255.
xTrain, xTest = np.expand_dims(xTrain, axis=3), np.expand_dims(xTest, axis=3)
yTrainOne = tf.one_hot(yTrain, depth=10).numpy()
yTestOne = tf.one_hot(yTest, depth=10).numpy()


def conv3x3(channels, stride=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels, kernel, strides=stride, padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer())


class ResnetBlock(keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.channels = channels
        self.strides = strides
        self.residualPath = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.downConv = conv3x3(channels, strides, kernel=(1, 1))
            self.downBn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residualPath:
            residual = self.downBn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.downConv(residual)

        x = x + residual
        return x


class ResNet(keras.Model):
    def __init__(self, blockList, numbClasses, initalFilters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.numBloacks = len(blockList)
        self.blockList = blockList

        self.inChannels = initalFilters
        self.outChannels = initalFilters
        self.convInital = conv3x3(self.outChannels)

        self.blocks = keras.models.Sequential(name='dynmic-blocks')

        for blockId in range(len(blockList)):
            for layerId in range(blockList[blockId]):
                if blockId != 0 and layerId == 0:
                    block = ResnetBlock(self.outChannels, strides=2, residual_path=True)
                else:
                    if self.inChannels != self.outChannels:
                        residualPath = True
                    else:
                        residualPath = False
                    block = ResnetBlock(self.outChannels, residual_path=residualPath)
                self.inChannels = self.outChannels
                self.blocks.add(block)
            self.outChannels *= 2

        self.finalBn = keras.layers.BatchNormalization()
        self.avgPool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(numbClasses)

    def call(self, inputs, training=None):
        out = self.convInital(inputs)
        out = self.blocks(out, training=training)
        out = self.finalBn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avgPool(out)
        out = self.fc(out)
        return out


def main():
    numClasses = 10
    batchSize = 32
    epochs = 1

    model = ResNet([2, 2, 2], numClasses)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    model.fit(xTrain, yTrainOne, batch_size=batchSize, epochs=epochs, validation_data=(xTest, yTestOne), verbose=1)
    scores = model.evaluate(xTest, yTestOne, batchSize, verbose=1)
    print("Accuracy : {0}".format(scores))

if __name__=='__main__':
    main()