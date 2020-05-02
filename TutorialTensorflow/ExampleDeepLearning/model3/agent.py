import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN

tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

# hyperparameter
maxEpisode = 10000
targetUpdateInterval = 1000  # target train frame
trainInterval = 4  # train frame
observe = 100  # start frame

# game parameter
numAction = 3
screenWidth = 6
screenHeight = 10


def train():
    print('뇌세포 꺠우는 중..')
    sess = tf.Session()

    game = Game(screenWidth, screenHeight, show_game=False)
    brain = DQN(sess, screenWidth, screenHeight, numAction)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summaryMerged = tf.summary.merge_all()

    brain.updateTargetNetwork()

    timeStep = 0
    totalRewardList = []

    for episode in range(maxEpisode):
        terminal = False
        totalReward = 0
        epsilon = 1.0

        state = game.reset()
        brain.initState(state)

        while not terminal:
            if np.random.rand() < epsilon:
                action = random.randrange(numAction)

            else:
                action = brain.getAction()
            if episode > observe:
                epsilon -= 1 / 1000

            state, reward, terminal = game.step(action)
            totalReward += reward
            brain.remember(state, action, reward, terminal)

            if timeStep > observe and timeStep % trainInterval == 0:
                brain.train()
            if timeStep % targetUpdateInterval == 0:
                brain.updateTargetNetwork()

            timeStep += 1

        totalRewardList.append(totalReward)

        if episode % 10 == 0:
            summary = sess.run(summaryMerged, feed_dict={rewards: totalRewardList})
            writer.add_summary(summary, timeStep)
        if episode % 100 == 99:
            print("게임횟수 : {0}, 점수 : {1:.4f}".format(episode+1, totalReward))
            saver.save(sess, './model/dqn.ckpt', global_step=timeStep)


def replay():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(screenWidth, screenHeight, show_game=True)
    brain = DQN(sess, screenWidth, screenHeight, numAction)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(maxEpisode):
        terminal = False
        totalReward = 0

        state = game.reset()
        brain.initState(state)

        while not terminal:
            action = brain.getAction()
            state, reward, terminal = game.step(action)
            totalReward += reward
            time.sleep(0.3)
        print("게임횟수 : {0}, 점수 : {1}".format(episode + 1, totalReward))


def main(_):
    if FLAGS.train:
        train()
    else:
        replay()

if __name__=='__main__':
    tf.app.run()