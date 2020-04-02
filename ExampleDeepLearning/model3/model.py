import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN:
    replayMemory = 10000
    batchSize = 32
    gamma = 0.99
    stateLen = 4

    def __init__(self, session, width, height, nAction):
        self.session = session
        self.nAction = nAction
        self.width = width
        self.height = height
        self.memory = deque()
        self.state = None

        self.inputX = tf.placeholder(tf.float32, [None, width, height, self.stateLen])
        self.inputA = tf.placeholder(tf.int64, [None])
        self.inputY = tf.placeholder(tf.float32, [None])

        self.Q = self._buildNetwork('main')
        self.cost, self.trainOp = self._buildOp()

        self.targetQ = self._buildNetwork('target')

    def _buildNetwork(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.inputX, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            Q = tf.layers.dense(model, self.nAction, activation=None)
        return Q

    def _buildOp(self):
        oneHot = tf.one_hot(self.inputA, self.nAction, 1.0, 0.0)
        QValue = tf.reduce_sum(tf.multiply(self.Q, oneHot), axis=1)
        cost = tf.reduce_mean(tf.square(self.inputY - QValue))
        trainOp = tf.train.AdamOptimizer(1e-6).minimize(cost)
        return cost, trainOp

    def updateTargetNetwork(self):
        copyOp = []
        mainVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        targetVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for mainVar, targetVar in zip(mainVars, targetVars):
            copyOp.append(targetVar.assign(mainVar.value()))
        self.session.run(copyOp)

    def getAction(self):
        QValue = self.session.run(self.Q, feed_dict={self.inputX: [self.state]})
        action = np.argmax(QValue[0])
        return action

    def train(self):
        state, nextState, action, reward, terminal = self._sampleMemory()
        targetQValue = self.session.run(self.targetQ, feed_dict={self.inputX: nextState})

        Y = []
        for i in range(self.batchSize):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.gamma * np.max(targetQValue[i]))
        self.session.run(self.trainOp, feed_dict={self.inputX: state, self.inputA: action, self.inputY: Y})

    def initState(self, state):
        state = [state for _ in range(self.stateLen)]
        self.state = np.stack(state, axis=2)

    def remember(self, state, action, reward, terminal):
        nextState = np.reshape(state, (self.width, self.height, 1))
        nextState = np.append(self.state[:, :, 1:], nextState, axis=2)
        self.memory.append((self.state, nextState, action, reward, terminal))

        if len(self.memory) > self.replayMemory:
            self.memory.popleft()
        self.state = nextState

    def _sampleMemory(self):
        sampleMemory = random.sample(self.memory, self.batchSize)
        state = [memory[0] for memory in sampleMemory]
        nextState = [memory[1] for memory in sampleMemory]
        action = [memory[2] for memory in sampleMemory]
        reward = [memory[3] for memory in sampleMemory]
        terminal = [memory[4] for memory in sampleMemory]

        return state, nextState, action, reward, terminal
