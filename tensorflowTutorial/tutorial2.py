import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='output')
Y = tf.placeholder(tf.float32, [None, 10], name="input")

with tf.name_scope("Layer1"):
    W1 = tf.Variable(initial_value=tf.random_normal([3, 3, 1, 32], stddev=0.01), name="W1")
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(initial_value=tf.random_normal([256, 256], stddev=0.01), name='W2')
    L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding="SAME")
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    L2 = tf.nn.dropout(L2, keep_prob)

with tf.name_scope("Layer3"):
    W3 = tf.Variable(initial_value=tf.random_normal([256, 10], stddev=0.01), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope("optimizer"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val
    print('Epoch: {0}, Avg.cost={1:.4f}'.format(epoch + 1, total_cost / total_batch))

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 : {0}".format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})))
sess.close()
