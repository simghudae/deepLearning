import tensorflow as tf
#tensorflow의 기초
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
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
y_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]])

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
# counting global step
global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 3], -1., 1.), name='W1')
    b1 = tf.Variable(tf.zeros([3]), name='b1')
    L1 = tf.add(tf.matmul(X, W1), b1)
    L1 = tf.nn.relu(L1)

    tf.summary.histogram("X", X)
    tf.summary.histogram("Weights1", W1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([3, 3], -1., 1.), name='W2')
    b2 = tf.Variable(tf.zeros([3]), name='b2')
    L2 = tf.add(tf.matmul(L1, W2), b2)

    tf.summary.histogram("Weights2", W2)
    tf.summary.histogram("L2", L2)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=L2))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

    # tensor board
    tf.summary.scalar('cost', cost)  # 값이 한개인 tensor를 수집하는 방법


# create session
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())


# restore tensor
ckpt = tf.train.get_checkpoint_state('.\\model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
#tensor board1
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    #tensor board2
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))
# save tensor
saver.save(sess, './model/dnn.ckpt', global_step=global_step)
sess.close()

#tensorboard --logdir=./logs

