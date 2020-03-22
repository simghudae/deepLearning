import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("./mnist/data", one_hot=True)
# tensorboard --logdir=./logs

learningRate = 0.0001
totalEpoch = 20
batchSize = 100

graphModel1 = tf.Graph()
with graphModel1.as_default():
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")

    tf.summary.histogram("X", X)
    isTraining = tf.placeholder(tf.bool)
    globalStep = tf.Variable(0, trainable=False)

    with tf.variable_scope('Case1') as scope:
        with tf.name_scope('Layer1'):
            L1 = tf.layers.conv2d(X, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, name="L1",
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            L1 = tf.layers.max_pooling2d(L1, pool_size=[2, 2], strides=[2, 2], padding="SAME")
            L1 = tf.layers.batch_normalization(L1, training=isTraining)

        with tf.name_scope('Layer2'):
            L2 = tf.layers.conv2d(L1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, name="L2",
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            L2 = tf.layers.max_pooling2d(L2, pool_size=[2, 2], strides=[2, 2], padding="SAME")
            L2 = tf.layers.batch_normalization(L2, training=isTraining)

        with tf.name_scope('Layer3'):
            L2 = tf.contrib.layers.flatten(L2)
            L3 = tf.layers.dense(L2, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            L3 = tf.layers.dropout(L3, rate=0.8, training=isTraining)

        with tf.name_scope('Layer4'):
            model = tf.layers.dense(L3, units=10, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope('Optimizer'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
            optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost, global_step=globalStep)
            tf.summary.scalar("cost", cost)

        with tf.name_scope('Testing'):
            isCorrect = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

with tf.Session(graph=graphModel1) as sess:
    saver = tf.train.Saver(tf.global_variables())
    # print(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('.\\model1')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs1', sess.graph)

    totalBatch = mnist.train.num_examples // batchSize
    for epoch in range(totalEpoch):
        totalError = 0
        for _ in range(totalBatch):
            batchXs, batchYs = mnist.train.next_batch(batchSize)
            batchXs = batchXs.reshape(-1, 28, 28, 1)
            _cost, _ = sess.run([cost, optimizer], feed_dict={X: batchXs, Y: batchYs, isTraining: True})
            totalError += _cost
        print("epoch : {0}, error : {1:.3f}".format(epoch + 1, totalError / totalBatch))
        summary = sess.run(merged, feed_dict={X: batchXs.reshape(-1, 28, 28, 1), Y: batchYs, isTraining: False})
        writer.add_summary(summary, global_step=sess.run(globalStep))

    print("accuracy : {0}".format(sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, isTraining: False})))
    saver.save(sess, "./model1/cnn.ckpt", global_step=globalStep)
