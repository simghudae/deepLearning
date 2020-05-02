import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setting Hyperparameter
valdationSize = 10000
dropOutRate = 0.5
learningRate = 0.001
batchSize = 64
maxEpoch = 10
weightDecay = 0.001

# Preprosseing Dataset
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, xTest = tf.cast(xTrain, tf.float32) / 255., tf.cast(xTest, tf.float32) / 255.
yTrain, yTest = tf.cast(tf.one_hot(yTrain, depth=10), tf.float64), tf.cast(tf.one_hot(yTest, depth=10), tf.float64)

xVal, xTrain = xTrain[:valdationSize], xTrain[valdationSize:]
yVal, yTrain = yTrain[:valdationSize], yTrain[valdationSize:]

# Create Model
cnnModel = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropOutRate),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropOutRate),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='sigmoid', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropOutRate),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay))
])

# Train/Test Process
optimizerFtn = tf.keras.optimizers.Adam(learning_rate=learningRate)
lossFtn = tf.losses.CategoricalCrossentropy()
cnnModel.compile(optimizer=optimizerFtn, loss=lossFtn, metrics=['accuracy'])
history = cnnModel.fit(x=xTrain, y=yTrain, batch_size=batchSize, epochs=maxEpoch, verbose=2, validation_data=(xVal, yVal),
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)])
result = cnnModel.evaluate(x=xTest, y=yTest, verbose=2)

# Visualization
import matplotlib.pyplot as plt
def plotErrorAcc(history):
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(history.epoch, history.history['accuracy'], label='train_accuracy')
    ax1.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(history.epoch, history.history['loss'], label='train_loss')
    ax2.plot(history.epoch, history.history['val_loss'], label='val_loss')
    ax2.legend()


plotErrorAcc(history)