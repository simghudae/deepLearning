import tensorflow as tf
import numpy as np

tf.Tensor([4,6], shape=(2,), dtype=tf.int32)

abc = np.ones([3,3])
tensor = tf.multiply(abc, 42)
print(tensor)