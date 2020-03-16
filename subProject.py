import tensorflow as tf

# rank = 차원수, shape = 차원의 요소
# ex rank=0 : 스칼라, rank=1 : vector, rank=2 : matrix ...
# type = string, float, int
# tensor = Graph 생성 + Graph 실행
hello = tf.constant('Hello, Tensorflow!')
print(hello)

# placeholder : 입력값을 나중에 받기 위해 사용하는 매개변수
# Variable : 학습결과를 갱신하기 위해 사용하는 변수
X = tf.placeholder(tf.float32, [None, 3], name='X')
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))
expr = tf.matmul(X, W) + b  # 행렬곱은 tf.matmul로 실시해야함
xData = [[1, 2, 3], [4, 5, 6]]

# graph의 실행은 session에서 실시하며, run method을 이용한다
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(hello))
print(sess.run(W))  # Session run으로 variable안의 값을 확인 가능
print(sess.run(expr, feed_dict={X: xData}))
sess.close()

# python의  with 기능을 통한 session block의 생성방법
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer)

# ---------------------------

import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
y_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)  # activate ftn
L = tf.nn.softmax(L)  # sum =1 normalize

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(L), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

