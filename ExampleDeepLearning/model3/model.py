import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28,)),
    keras.layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2, 2,), (2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(10)])

computeLoss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
computeAccuracy = keras.metrics.SparseCategoricalAccuracy()
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.5)


def mnist_dataset():
    (xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()
    xTrain, xTest = xTrain / np.float32(255), xTest / np.float32(255)
    yTrain, yTest = yTrain.astype(np.int64), yTest.astype(np.int64)
    trainDataSet = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
    testDataSet = tf.data.Dataset.from_tensor_slices((xTest, yTest))
    return trainDataSet, testDataSet


trainDs, testDs = mnist_dataset()
trainDs = trainDs.shuffle(60000).batch(100)
testDs = testDs.batch(100)

def train(model, optimizer, dataSet, epoch):
    avgLoss = keras.metrics.Mean('loss', dtype=tf.float32)
    for images, labels in dataSet:
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = computeLoss(labels, logits)
            computeAccuracy(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        avgLoss(loss)

    summary_ops_v2.scalar('loss', avgLoss.result(), step=epoch)
    summary_ops_v2.scalar('accuracy', computeAccuracy.result(), step=epoch)
    print("epoch : {0}, loss : {1:.3f}, acc : {2:.3f}".format(epoch+1, avgLoss.result().numpy(), computeAccuracy.result().numpy()))
    avgLoss.reset_states()
    computeAccuracy.reset_states()

def test(model, dataSet):
    avgLoss = keras.metrics.Mean('loss', dtype=tf.float32)
    for (images, labels) in dataSet:
        logits = model(images, training=False)
        avgLoss(computeLoss(labels, logits))
        computeAccuracy(labels, logits)
    print("loss : {0:.4f}, Acc : {1:.4f}".format(avgLoss.result(), computeAccuracy.result()))

def applyClean(modelDIR):
    if tf.io.gfile.exists(modelDIR):
        print('Removing existing model dir :{}'.format(modelDIR))
        tf.io.gfile.rmtree(modelDIR)

modelDIR = './tmp/'
# applyClean(modelDIR)

checkpointDir = os.path.join(modelDIR, 'checkpoints')
checkpointPrefix = os.path.join(checkpointDir, 'ckpt')
ckeckpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckeckpoint.restore(tf.train.latest_checkpoint(checkpointDir))


totalEpoch = 10
for i in range(totalEpoch):
    train(model, optimizer, trainDs, i)
    ckeckpoint.save(checkpointPrefix)

exportPath = os.path.join(modelDIR, 'export')
tf.saved_model.save(model, modelDIR)

