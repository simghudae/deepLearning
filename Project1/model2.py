import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# tensorboard --logdir=./logs
batchSize = 100
totalEpoch = 20
learningRate = 0.00001
n_input = 28 * 28
n_hidden = 256
n_noise = 128
n_class = 10


def getNoise(batchSize, n_noise):
    return np.random.uniform(-1, 1, [batchSize, n_noise])


graphModel2 = tf.Graph()
with graphModel2.as_default():
    with tf.name_scope("DefaultVariable"):
        X = tf.placeholder(tf.float32, [None, n_input])
        Y = tf.placeholder(tf.float32, [None, n_class])
        Z = tf.placeholder(tf.float32, [None, n_noise])
        isTraining = tf.placeholder(tf.bool)
        globalStep = tf.Variable(0, trainable=False)

    with tf.name_scope("CreateModel"):
        def generator(noise, labels, isTraining):
            with tf.variable_scope('generator', initializer=tf.contrib.layers.xavier_initializer()) as generate:
                inputLayer = tf.concat([noise, labels], 1)
                hiddenLayer1 = tf.layers.dense(inputLayer, n_hidden, activation=tf.nn.relu)
                hiddenLayer1 = tf.layers.batch_normalization(hiddenLayer1, training=isTraining)
                hiddenLayer2 = tf.layers.dense(hiddenLayer1, n_input, activation=tf.nn.sigmoid)
                return hiddenLayer2


        def discriminator(input, labels, isTraining, reUse=False):
            with tf.variable_scope('discriminator', initializer=tf.contrib.layers.xavier_initializer()) as discriminate:
                if reUse: discriminate.reuse_variables()
                inputs = tf.concat([input, labels], 1)
                hiddenLayer1 = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
                hiddenLayer1 = tf.layers.batch_normalization(hiddenLayer1, training=isTraining)
                hiddenLayer2 = tf.layers.dense(hiddenLayer1, 1, activation=tf.nn.sigmoid)
                return hiddenLayer2


        generateModel = generator(Z, Y, isTraining)
        discriminateGene = discriminator(generateModel, Y, isTraining)
        discriminateReal = discriminator(X, Y, isTraining, True)

    with tf.name_scope("Optimize"):
        lossDiscriminateGene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminateGene, labels=tf.zeros_like(discriminateGene)))
        lossDiscriminateReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminateReal, labels=tf.ones_like(discriminateReal)))
        lossDiscriminate = lossDiscriminateGene + lossDiscriminateReal
        lossGenerate = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminateGene, labels=tf.ones_like(discriminateGene)))

        variableDiscriminate = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        variableGenerate = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        trainDiscriminate = tf.train.AdamOptimizer(learningRate).minimize(-lossDiscriminate, var_list=variableDiscriminate, global_step=globalStep)
        trainGenerate = tf.train.AdamOptimizer(learningRate).minimize(-lossGenerate, var_list=variableGenerate, global_step=globalStep)
        tf.summary.scalar("lossDiscriminate", lossDiscriminate)
        tf.summary.scalar("lossGenerate", lossGenerate)

totalBatch = mnist.train.num_examples // batchSize
lossDisc, lossGene = 0, 0

with tf.Session(graph=graphModel2) as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('.\\model2')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs2', sess.graph)

    for epoch in range(totalEpoch):
        errorGene, errorDisc = 0, 0
        for _ in range(totalBatch):
            trainXs, trainYs = mnist.train.next_batch(batchSize)
            trainNoise = getNoise(batchSize, n_noise)
            _errorGene, _ = sess.run([lossGenerate, trainGenerate], feed_dict={Z: trainNoise, Y: trainYs, isTraining: True})
            _errorDisc, _ = sess.run([lossDiscriminate, trainDiscriminate], feed_dict={X: trainXs, Y: trainYs, Z: trainNoise, isTraining: True})
            errorGene += _errorGene
            errorDisc += _errorDisc
        summary = sess.run(merged, feed_dict={X: trainXs, Y: trainYs, Z: trainNoise, isTraining: True})
        writer.add_summary(summary, global_step=sess.run(globalStep))

        print("Epoch : {0}, Generator : {1:.3f}, Discriminator : {2:.3f}".format(epoch + 1, errorGene / totalBatch, errorDisc / totalBatch))
    saver.save(sess, "./model2/gan.ckpt", global_step=globalStep)
