import tensorflow as tf
import os

# setting hyperparameter
batchSize = 32
weightDecay = 0.001
dropoutRate = 0.3
learningRate = 0.001
imageSize = (128, 128)
maxEpoch = 10

# download Dog/Cat images
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# setting directory
trainDir = os.path.join(PATH, 'train')
validationDir = os.path.join(PATH, 'validation')
trainCatsDir = os.path.join(trainDir, 'cats')
trainDogsDir = os.path.join(trainDir, 'dogs')
validationCatsDir = os.path.join(validationDir, 'cats')
validationDogsDir = os.path.join(validationDir, 'dogs')

lenTrain = len(os.listdir(trainCatsDir)) + len(os.listdir(trainDogsDir))
lenVal = len(os.listdir(validationCatsDir)) + len(os.listdir(validationDogsDir))

# preprosseing image
trainImageGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255., rotation_range=45, zoom_range=0.3, horizontal_flip=True, width_shift_range=0.3,
                                                                      height_shift_range=0.3)
valImageGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
xyTrain = trainImageGenerator.flow_from_directory(directory=trainDir, target_size=imageSize, class_mode='binary', batch_size=batchSize, shuffle=True)
xyVal = valImageGenerator.flow_from_directory(directory=validationDir, class_mode='binary', target_size=imageSize)


# create model
cnnModel = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay),
                           input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropoutRate),
    tf.keras.layers.MaxPool2D(pool_size=(4, 4)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropoutRate),
    tf.keras.layers.MaxPool2D(pool_size=(4, 4)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropoutRate),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')
])

# train, test
cnnModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
history = cnnModel.fit_generator(generator=xyTrain, epochs=maxEpoch, steps_per_epoch=lenTrain // batchSize, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)],
                                 validation_data=xyVal, validation_steps=lenVal // batchSize)

# visualization
import matplotlib.pyplot as plt

def plotErrorACC(history):
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(history.epoch, history.history['loss'], label='train_loss')
    ax1.plot(history.epoch, history, history['val_loss'], label='val_loss')

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(history.epoch, history.history['accuracy'], label='train_accuracy')
    ax2.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')

plotErrorACC(history)