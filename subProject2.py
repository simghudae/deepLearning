import tensorflow as tf
x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# 입력 텐서 x에 대한 z의 도함수
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    print(dz_dx[i][j].numpy())


    assert dz_dx[i][j].numpy() == 8.0