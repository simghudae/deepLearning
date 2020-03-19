import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

learning_rate = 0.01
training_epoch = 20
batch_size = 100
n_hidden = 256
n_input = 28 * 28

X = tf.placeholder(tf.float32, [None, n_input])

W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

loss = tf.reduce_mean(tf.square(X - decoder))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch):
    total_error = 0
    for batch in range(total_batch):
        batch_xs,_ = mnist.train.next_batch(batch_size)
        error, _ = sess.run([loss, optimizer], feed_dict={X: batch_xs})
        total_error += error
    print("Epoch : {0}, Error :{1:.3f}".format(epoch+1, total_error/total_batch))

sample_size =10
samples = sess.run(decoder, feed_dict={X:mnist.test.images[:sample_size]})
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][1].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    ax[1][i].imshow(np.reshape(samples[i], (28,28)))
sess.close()

#------------------------------

