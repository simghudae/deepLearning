import tensorflow as tf
from tensorflow import keras

(xs, ys), _ = keras.datasets.mnist.load_data()
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs, ys))
db = db.batch(32).repeat(10)

network = keras.Sequential([keras.layers.Dense(256, activation='relu'),
                            keras.layers.Dense(256, activation='relu'),
                            keras.layers.Dense(256, activation='relu'),
                            keras.layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
optimizer = keras.optimizers.SGD(lr=0.01)
accMeter = keras.metrics.Accuracy()

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28 * 28))
        predict = network(x)
        yOnehot = tf.one_hot(y, depth=10)
        loss = tf.square(predict - yOnehot)
        loss = tf.reduce_sum(loss) / 32
    accMeter.update_state(tf.argmax(predict, axis=1), y)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 200 == 0:
        print("step : {0}, loss :{1:.3f}, acc : {2:.3f}".format(step, loss, accMeter.result().numpy()))
        accMeter.reset_states()
