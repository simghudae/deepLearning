import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.zeros([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = mnist.train.num_examples // batch_size

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val
    print('Epoch : {0}, Avg cost ={1:.3f}'.format(epoch + 1, total_cost / total_batch))

# -------------------------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np

seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
dicLen = ord('z') - ord('a') + 1


def makeBatch(seqData):
    inputBatch, targetBatch = [], []
    for seq in seqData:
        input = [ord(i) - ord('a') for i in seq[:-1]]
        target = ord(seq[-1]) - ord('a')
        inputBatch.append(np.eye(dicLen)[input])
        targetBatch.append(target)
    return inputBatch, targetBatch


learningRate = 0.001
nHidden = 128
totalEpoch = 30

nStep, nInput, nClass = 3, dicLen, dicLen

X = tf.placeholder(tf.float32, [None, nStep, nInput])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([nHidden, nClass]))
b = tf.Variable(tf.random_normal([nClass]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(nHidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(nHidden)

multiCell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multiCell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inputBatch, targetBatch = makeBatch(seq_data)

    for epoch in range(totalEpoch):
        _, loss = sess.run([optimizer, cost], feed_dict={X: inputBatch, Y: targetBatch})
        print('Epoch : {0}, Cost : {1:.5f}'.format(epoch + 1, loss))

# ----------------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np

charArr = [c for c in 'SEP단어나무놀이소녀키스사랑']
numDic = {n: i for i, n in enumerate(charArr)}
dicLen = len(numDic)+26

seqData = [['word', '단어'], ['wood', '나무'], ['game', '놀이'], ['girl', '소녀'], ['kiss', '키스'], ['love', '사랑']]


def makeBatch(seqData):
    inputBatch, outputBatch, targetBatch = [], [], []
    for seq in seqData:
        inputs = [ord(i) - ord('a') for i in seq[0]]
        outputs = [numDic[n]+26 for n in ('S' + seq[1])]
        targets = [numDic[n]+26 for n in (seq[1] + 'E')]
        inputBatch.append(np.eye(dicLen)[inputs])
        outputBatch.append(np.eye(dicLen)[outputs])
        targetBatch.append(targets)
    return inputBatch, outputBatch, targetBatch


learningRate = 0.001
nHidden = 128
totalEpoch = 100

nClass, nInput = dicLen,dicLen

encInput = tf.placeholder(tf.float32, [None, None, nInput])
decInput = tf.placeholder(tf.float32, [None, None, nInput])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    encCell = tf.nn.rnn_cell.BasicRNNCell(nHidden)
    encCell = tf.nn.rnn_cell.DropoutWrapper(encCell, output_keep_prob=0.5)
    outputs, encStates = tf.nn.dynamic_rnn(encCell, encInput, dtype=tf.float32)

with tf.variable_scope('decode'):
    decCell = tf.nn.rnn_cell.BasicRNNCell(nHidden)
    decCell = tf.nn.rnn_cell.DropoutWrapper(decCell, output_keep_prob=0.5)
    outputs, decStates = tf.nn.dynamic_rnn(decCell, decInput, initial_state=encStates, dtype=tf.float32)

model = tf.layers.dense(outputs, nClass, activation=None)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inputBatch, outputBatch, targetBatch = makeBatch(seqData)
    for epoch in range(totalEpoch):
        _, loss = sess.run([optimizer, cost], feed_dict={encInput: inputBatch, decInput: outputBatch, targets: targetBatch})
        print("Epoch : {0}, Cost : {1:.5f}".format(epoch + 1, loss))
