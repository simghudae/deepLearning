import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

# class Generator(keras.Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.n_f = 512
#         self.n_k = 4
#
#         # input z vector is [None, 100]
#         self.dense1 = keras.layers.Dense(3 * 3 * self.n_f)
#         self.conv2 = keras.layers.Conv2DTranspose(self.n_f // 2, 3, 2, 'valid')
#         self.bn2 = keras.layers.BatchNormalization()
#         self.conv3 = keras.layers.Conv2DTranspose(self.n_f // 4, self.n_k, 2, 'same')
#         self.bn3 = keras.layers.BatchNormalization()
#         self.conv4 = keras.layers.Conv2DTranspose(1, self.n_k, 2, 'same')
#         return
#
#     def call(self, inputs, training=None):
#         # [b, 100] => [b, 3, 3, 512]
#         x = tf.nn.leaky_relu(tf.reshape(self.dense1(inputs), shape=[-1, 3, 3, self.n_f]))
#         x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
#         x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
#         x = tf.tanh(self.conv4(x))
#         return x

generator = tf.keras.Sequential([
    tf.keras.layers.Reshape(input_shape=[49], target_shape=[7, 7, 1]),
    tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
])

# class Discriminator(keras.Model):
#
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.n_f = 64
#         self.n_k = 4
#
#         # input image is [-1, 28, 28, 1]
#         self.conv1 = keras.layers.Conv2D(self.n_f, self.n_k, 2, 'same')
#         self.conv2 = keras.layers.Conv2D(self.n_f * 2, self.n_k, 2, 'same')
#         self.bn2 = keras.layers.BatchNormalization()
#         self.conv3 = keras.layers.Conv2D(self.n_f * 4, self.n_k, 2, 'same')
#         self.bn3 = keras.layers.BatchNormalization()
#         self.flatten4 = keras.layers.Flatten()
#         self.dense4 = keras.layers.Dense(1)
#         return
#
#     def call(self, inputs, training=None):
#         x = tf.nn.leaky_relu(self.conv1(inputs))
#         x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
#         x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
#         x = self.dense4(self.flatten4(x))
#         return x


discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 3], activation='relu', padding='same'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 3], activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='softmax')
])


# shorten sigmoid cross entropy loss calculation
def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.ones_like(logits) * (1.0 - smooth)))


def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf.zeros_like(logits) * (1.0 - smooth)))


def d_loss_fn(generator, discriminator, input_noise, real_image, is_trainig, batchSize):
    fake_image = generator(input_noise, is_trainig)
    logitReal = discriminator(real_image, is_trainig)
    logitFake = discriminator(fake_image, is_trainig)

    outputZero = np.zeros([batchSize, 1])
    outputOne = np.ones([batchSize, 1])

    d_loss_real = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=outputOne, y_pred=logitFake))
    d_loss_fake = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=outputZero, y_pred=logitReal))
    loss = d_loss_real + d_loss_fake
    return loss


def g_loss_fn(generator, discriminator, input_noise, is_trainig):
    fake_image = generator(input_noise, is_trainig)
    d_fake_logits = discriminator(fake_image, is_trainig)
    loss = celoss_ones(d_fake_logits, smooth=0.1)
    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 49
    epochs = 3000000
    batch_size = 128
    learning_rate = 0.0002
    is_training = True

    # for validation purpose
    assets_dir = './images'
    if not os.path.isdir(assets_dir):
        os.makedirs(assets_dir)
    val_block_size = 10
    val_size = val_block_size * val_block_size

    # load mnist data
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batch_size * 4).batch(batch_size).repeat()
    db_iter = iter(db)
    inputs_shape = [-1, 28, 28, 1]

    # create generator & discriminator
    # generator = Generator()
    generator.build(input_shape=(batch_size, z_dim))
    generator.summary()
    # discriminator = Discriminator()
    discriminator.build(input_shape=(batch_size, 28, 28, 1))
    discriminator.summary()

    # prepare optimizer
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):

        # no need labels
        batch_x = next(db_iter)

        # rescale images to -1 ~ 1
        batch_x = tf.reshape(batch_x, shape=inputs_shape)
        # -1 - 1
        batch_x = batch_x * 2.0 - 1.0

        # Sample random noise for G
        batch_z = tf.random.uniform(shape=[batch_size, z_dim], minval=-1., maxval=1.)

        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training, batch_size)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd loss:', float(d_loss), 'g loss:', float(g_loss))

            # validation results at every epoch
            val_z = np.random.uniform(-1, 1, size=(val_size, z_dim))
            fake_image = generator(val_z, training=False)
            image_fn = os.path.join('images', 'gan-val-{:03d}.png'.format(epoch + 1))
            save_result(fake_image.numpy(), val_block_size, image_fn, color_mode='L')