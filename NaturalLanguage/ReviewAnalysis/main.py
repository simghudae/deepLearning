import tensorflow_datasets as tdf
import tensorflow as tf

tdf.disable_progress_bar()

# donwload Review
# !pip install wget
# !pip install konlpy
# !python -m wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
# !python -m wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt

# hyperparameter
valdationSize = 10000
maxLength = 20
embeddingLength = 128
weightDecay = 0.001
learningRate = 0.001
maxEpoch = 20
batchSize = 32

# Step0 : Loading Data
import random

with open('ratings_train.txt', encoding='UTF8') as f:
    rawTrain = f.readlines()

with open('ratings_test.txt', encoding='UTF8') as f:
    rawTest = f.readlines()

rawTrain, rawTest = rawTrain[1:], rawTest[1:]
random.shuffle(rawTrain), random.shuffle(rawTest)

xTrain, yTrain = [train.split("\t")[1] for train in rawTrain], [int(train.split("\t")[2].split("\n")[0]) for train in rawTrain]
xTest, yTest = [test.split("\t")[1] for test in rawTest[1:]], [int(test.split("\t")[2].split("\n")[0]) for test in rawTest[1:]]

xTrain, xVal = xTrain[valdationSize:], xTrain[:valdationSize]
yTrain, yVal = yTrain[valdationSize:], yTrain[:valdationSize]

print("Seperate data set is done")


# Step1 : Parsing and Make Dictionary
from konlpy.tag import Okt
okt = Okt()


def tokenize(lines):
    return [pos[0] for pos in okt.pos(lines)]


vocabDict = {"[PAD]": 0, "[OOV]": 1}
i = 2

xTrainToken, yTrainOnehot = [], []
for lines, label in zip(xTrain, yTrain):
    realTokens = tokenize(lines)
    tokens = [vocabDict['[PAD]'] for _ in range(maxLength)]
    for k, token in enumerate(realTokens):
        if token not in vocabDict.keys():  # make word dictionary
            vocabDict[token] = i
            i += 1
        if k < maxLength:  # make same length input type
            tokens[k] = vocabDict[token]
    xTrainToken.append(tokens)

    if label == 0:
        yTrainOnehot.append([1, 0])
    else:
        yTrainOnehot.append([0, 1])

    if len(xTrainToken) % 10000 == 0:
        print("Training Tokenize {0} / {1} is done".format(len(xTrainToken), len(xTrain)))

xValToken, yValOnehot = [], []
for lines, label in zip(xVal, yVal):
    realTokens = tokenize(lines)
    tokens = [vocabDict['[PAD]'] for _ in range(maxLength)]
    for k, token in enumerate(realTokens):
        if k >= maxLength:
            break
        if token in vocabDict.keys():  # make same input lenth
            tokens[k] = vocabDict[token]
        else:
            tokens[k] = vocabDict['[OOV]']
    xValToken.append(tokens)
    if label == 0:
        yValOnehot.append([1, 0])
    else:
        yValOnehot.append([0, 1])

    if len(xValToken) % 10000 == 0:
        print("Validation Tokenize {0} / {1} is done".format(len(xValToken), len(xVal)))

xTestToken, yTestOnehot = [], []
for lines, label in zip(xTest, yTest):
    realTokens = tokenize(lines)
    tokens = [vocabDict['[PAD]'] for _ in range(maxLength)]
    for k, token in enumerate(realTokens):
        if k >= maxLength:
            break
        if token in vocabDict.keys():  # make same input length
            tokens[k] = vocabDict[token]
        else:
            tokens[k] = vocabDict['[OOV]']
    xTestToken.append(tokens)
    if label == 0:
        yTestOnehot.append([1, 0])
    else:
        yTestOnehot.append([0, 1])

    if len(xTestToken) % 10000 == 0:
        print("Test Tokenize {0} / {1} is done".format(len(xTestToken), len(xTest)))


# Step2 : make Model

LSTMModel = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocabDict), output_dim=embeddingLength),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(weightDecay)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# LSTMModel.summary()

# Step3 : training Model
LSTMModel.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), metrics=['accuracy'])
history = LSTMModel.fit(xTrainToken, yTrainOnehot, batch_size=batchSize, epochs=maxEpoch, verbose=2, validation_data=(xValToken, yValOnehot),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)])
result = LSTMModel.evaluate(xTestToken, yTestOnehot, verbose=2)


# Step4 : Visualization
import matplotlib.pyplot as plt
def plotAccLoss(history):
    plt.figure(figsize=[12, 6])
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(history.epoch, history.history['loss'], label='loss')
    ax1.plot(history.epoch, history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(history.epoch, history.history['accuracy'], label='accuracy')
    ax2.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()


plotAccLoss(history)