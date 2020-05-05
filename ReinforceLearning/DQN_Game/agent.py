import tensorflow as tf
import numpy as np
import random


class DQNAgent:
    def __init__(self, screenHeight, screenWidth, nAction, learningRate, updateRate, maxIteration):
        # setting Hyperparameter
        self.screenHeight, self.screenWidth, self.nAction = screenHeight, screenWidth, nAction
        self.learningRate, self.updateRate, self.dataBase = learningRate, updateRate, []
        self.maxIteration = maxIteration

        # setting model
        self.screen = np.zeros([self.screenHeight, self.screenWidth])
        self.mainQ = self._buildNetwork2()
        self.targetQ = self._buildNetwork2()

    def _buildNetwork1(self):
        modelDQN = tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape=(self.screenHeight, self.screenWidth, 1),
                                    input_shape=(self.screenHeight, self.screenWidth)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_initializer='glorot_normal'),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_initializer='glorot_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(self.nAction, activation='softmax', kernel_initializer='glorot_normal')
        ])
        modelDQN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningRate),
                         loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return modelDQN

    def _buildNetwork2(self):
        modelDQN = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.nAction, activation='softmax', kernel_initializer='glorot_normal')
        ])
        modelDQN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningRate),
                         loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return modelDQN

    def updateTargetNetwork(self):
        mainWeight, targetWeight = np.array(self.mainQ.get_weights()), np.array(self.targetQ.get_weights())
        self.mainQ.set_weights(mainWeight * (1 - self.updateRate) + targetWeight * self.updateRate)

    def getAction(self, state, iteration, training=True, version='method1'):
        if version == 'method1':
            if training and random.uniform(0, 1) >= float(iteration / self.maxIteration):
                return random.randint(1, self.nAction)
            else:
                return np.argmax(self.mainQ.predict(x=state))


        elif version == 'method2':
            if training:
                getnoiseQvalue = ((self.maxIteration - iteration) / self.maxIteration) * np.random.uniform(
                    size=[1, self.nAction]) + (iteration / self.maxIteration) * self.mainQ.predict(x=state)
                return np.argmax(getnoiseQvalue)
            else:
                return np.argmax(self.mainQ.predict(x=state))
