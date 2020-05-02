import tensorflow as tf
import tensorflow_hub as hub

# setting Hyperparameter
batchSize = 64
learningRate = 0.001
maxEpoch = 20
imageShape = (224, 224)

# preprocessing Image
dataRoot = tf.keras.utils.get_file('flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)

imageGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
imageData = imageGenerator.flow_from_directory(directory=str(dataRoot), batch_size=batchSize, target_size=imageShape)

# make Model
classifierUrl = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
classifierModel = tf.keras.Sequential([
    hub.KerasLayer(classifierUrl, input_shape=imageShape+(3,), trainable=False),
    tf.keras.layers.Dense(imageData.num_classes, activation='softmax')
])

# train Model
classifierModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
classifierModel.fit_generator(generator=imageData, epochs=maxEpoch, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)])

#save Model
modelName = "myModel"
classifierModel.save(modelName, save_format='tf')

#reload Model
reloaded = tf.keras.models.load_model(modelName)
reloaded.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
reloaded.evaluate_generator(generator=imageData, verbose=2)