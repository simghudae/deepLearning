import numpy as np
import random

import ReinforceLearning.DQN_Game.agent as agent
import ReinforceLearning.DQN_Game.game as game

# setting Hyperparameter
screenHeight, screenWidth = 10, 10
nAction, maxEpoch, maxIteration, levelUp = 2, 100, 1000, 20

batchSize, learningRate, updateRate, gamma = 50, 0.00001, 0.1, 0.1
numStringAction = {0: 'left', 1: 'right'}
dataBase = []

# setting Agent / Environment
Env = game.Game(screenHeight, screenWidth, levelUp, controlUser=False, showGame=True)
Agent = agent.DQNAgent(screenHeight, screenWidth, nAction, learningRate, updateRate, maxEpoch)


def oneStep(Env, Agent, dataBase, epoch):
    _state, _parameter = Env.getState()
    _action = Agent.getAction(_state, epoch + 1)
    Env.setAction(numStringAction[_action])
    _Nstate, _Nparameter = Env.getState()
    _done, _reward = _parameter[0], _parameter[1]
    dataBase.append([_state, _Nstate, _action, _done, _reward])


def sampling(dataBase, batchSize):
    datas = random.sample(dataBase, batchSize)
    currStates = np.squeeze(np.array([data[0] for data in datas]))
    nextStates = np.squeeze(np.array([data[1] for data in datas]))
    actions = np.squeeze(np.array([data[2] for data in datas]))
    dones = np.squeeze(np.array([data[3] for data in datas]))
    rewards = np.squeeze(np.array([data[4] for data in datas]))
    dataBase = []
    return dataBase, currStates, nextStates, actions, dones, rewards


for epoch in range(maxEpoch):
    for iteration in range(maxIteration):
        oneStep(Env, Agent, dataBase, epoch)

        if len(dataBase) > batchSize:
            dataBase, currStates, nextStates, actions, dones, rewards = sampling(dataBase, batchSize)
            qValues = rewards + gamma * np.amax(Agent.mainQ.predict(x=nextStates), axis=1) * dones
            realValues = Agent.mainQ.predict(x=currStates)
            for index in range(len(qValues)):
                realValues[index][actions[index]] = qValues[index]
            Agent.targetQ.fit(x=currStates, y=realValues, verbose=2)
            Agent.updateTargetNetwork()

done = True
Env = game.Game(screenHeight, screenWidth, levelUp, controlUser=False, showGame=True)
while done:
    _state, _parameter = Env.getState()
    done = _parameter[0]
    Env.setAction(numStringAction[Agent.getAction(_state, maxEpoch)])