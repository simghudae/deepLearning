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

total_batch = mnist.train.num_examples//batch_size

for epoch in range(total_epoch):
    total_cost =0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys})
        total_cost += cost_val
    print('Epoch : {0}, Avg cost ={1:.3f}'.format(epoch+1, total_cost/total_batch))