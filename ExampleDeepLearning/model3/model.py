import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_dataset():
    (x, y), _ = keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_feature_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(1000)
    return ds


@tf.function
def prepare_mnist_feature_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10)])

optimizer = keras.optimizers.Adam()
train_ds = mnist_dataset()
loss, accuracy = 0.0, 0.0
for epoch in range(20):
    for step, (x, y) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = compute_loss(logits,y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        accuracy = compute_accuracy(logits, y)
        if step % 100 == 0:
            print("Epoch : {0}, Loss : {1:.4f}, Accuracy : {2:.4f}".format(epoch + 1, loss.numpy(), accuracy.numpy()))



