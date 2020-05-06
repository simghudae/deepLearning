import tensorflow as tf
import numpy as np
import os

# download Dog/Cat images
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# setting directory
trainDir = os.path.join(PATH, 'train')

# setting Hyperparameter
noiseSize, noiseInput = tuple([64]), (8, 8)
imageSize = (128, 128)
batchSize = 32
learningRate = 0.001
maxEpoch = 10

# preprosseing image
trainImageGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
xyTrain = trainImageGenerator.flow_from_directory(directory=trainDir, target_size=imageSize, class_mode='binary', batch_size=batchSize, shuffle=True)


# # setting Model
# generator = tf.keras.Sequential([
#     tf.keras.layers.Reshape(input_shape=noiseSize, target_shape=noiseInput + tuple([1])),
#     tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization()
# ])
#

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.reshape1 = tf.keras.layers.Reshape(input_shape=noiseSize, target_shape=noiseInput + tuple([1]))
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(4, 4), activation='relu')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(4, 4), activation='relu')
        self.batch2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        layer1 = self.reshape1(inputs)
        layer1 = self.conv1(layer1)
        layer2 = self.batch1(layer1)
        layer2 = self.conv2(layer2)
        output = self.batch2(layer2)
        return output


# discriminator = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1, activation='softmax')
# ])


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(128, 128, 3))
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.Flatten = tf.keras.layers.Flatten()
        self.Dense = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs):
        layer1 = self.conv1(inputs)
        layer1 = self.batch1(layer1)
        layer2 = self.conv2(layer1)
        layer2 = self.batch2(layer2)
        layer3 = self.Flatten(layer2)
        output = self.Dense(layer3)
        return output


discriminator = Discriminator()
discriminator.build(input_shape=(batchSize, 128, 128, 3))
generator = Generator()
generator.build(input_shape=(batchSize, noiseSize[0]))

# train, test
# generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
# discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

disOptimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
genOptimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)

fakeImage = generator.predict(np.random.normal(0, 1, [batchSize, noiseSize[0]]))
logitFake = discriminator.predict(fakeImage)
logitReal = discriminator.predict(xyTrain.next()[0])

# outputZero = np.zeros([batchSize, 1])
# outputOne = np.ones([batchSize, 1])


def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * (1.0 - smooth)))


def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits) * (1.0 - smooth)))


def disLossFn():
    lossFake = celoss_ones(logits=logitFake)
    lossReal = celoss_zeros(logits=logitReal)
    return lossFake + lossReal


# def disLossFn():
#     lossFake = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=outputOne, y_pred=logitFake))
#     lossReal = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=outputZero, y_pred=logitReal))
#     return lossFake + lossReal

with tf.GradientTape() as tape:
    disLoss = disLossFn()
grads = tape.gradient(disLoss, discriminator.trainable_variables)
disOptimizer.apply_gradients(zip(grads, discriminator.trainable_variables))


def g_loss_fn(generator, discriminator, input_noise, is_trainig):
    fake_image = generator(input_noise, is_trainig)
    d_fake_logits = discriminator(fake_image, is_trainig)
    loss = celoss_ones(d_fake_logits, smooth=0.1)
    return loss


def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * (1.0 - smooth)))


def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits) * (1.0 - smooth)))


def d_loss_fn(generator, discriminator, input_noise, real_image, is_trainig):
    fake_image = generator(input_noise, is_trainig)
    d_real_logits = discriminator(real_image, is_trainig)
    d_fake_logits = discriminator(fake_image, is_trainig)

    d_loss_real = celoss_ones(d_real_logits, smooth=0.1)
    d_loss_fake = celoss_zeros(d_fake_logits, smooth=0.0)
    loss = d_loss_real + d_loss_fake
    return loss