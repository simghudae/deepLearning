import numpy as np
import random

import ReinforceLearning.DQN_Game.agent as agent
import ReinforceLearning.DQN_Game.game as game


# setting Hyperparameter
screenHeight, screenWidth = 10, 10
nAction, maxEpoch, maxIteration, levelUp = 2, 100, 1000, 20
batchSize, learningRate, updateRate, gamma = 50, 0.00001, 0.1, 0.1


class DQN_main():
    def __init__(self, screenHeight, screenWidth, nAction, maxEpoch, maxIteration, levelUp, batchSize, learningRate, updateRate, gamma,showgaming=True):
        self.screenHeight, self.screenWidth = screenHeight, screenWidth
        self.nAction, self.maxEpoch, self.maxIteration, self.levelUpLine = nAction, maxEpoch, maxIteration, levelUp

        self.batchSize, self.learningRate, self.updateRate, self.gamma = batchSize, learningRate, updateRate, gamma
        self.numStringAction = {0: 'left', 1: 'right'}
        self.dataBase = []

        # setting Agent / Environment
        self.Env = game.Game(self.screenHeight, self.screenWidth, self.levelUpLine, controlUser=False, showGame=showgaming)
        self.Agent = agent.DQNAgent(self.screenHeight, self.screenWidth, self.nAction, self.learningRate, self.updateRate, self.maxIteration)


    def _oneStep(self, iteration):
        _state, _parameter = self.Env.getState() #_parameter = gameState, reward, step, gameNum
        _action = self.Agent.getAction(_state, iteration + 1)
        self.Env.setAction(self.numStringAction[_action])
        _Nstate, _Nparameter = self.Env.getState()
        _done, _reward = _parameter[0], _parameter[1]
        self.dataBase.append([_state, _Nstate, _action, _done, _reward])


    def _sampling(self):
        datas = random.sample(self.dataBase, self.batchSize)
        currStates = np.squeeze(np.array([data[0] for data in datas]))
        nextStates = np.squeeze(np.array([data[1] for data in datas]))
        actions = np.squeeze(np.array([data[2] for data in datas]))
        dones = np.squeeze(np.array([data[3] for data in datas]))
        rewards = np.squeeze(np.array([data[4] for data in datas]))
        self.dataBase = []
        return currStates, nextStates, actions, dones, rewards

    def training(self):
        for epoch in range(maxEpoch):
            for iteration in range(maxIteration):
                self._oneStep(iteration)

                if len(self.dataBase) > self.batchSize:
                    dataBase, currStates, nextStates, actions, dones, rewards = self._sampling()
                    qValues = rewards + gamma * np.amax(self.Agent.mainQ.predict(x=nextStates), axis=1) * dones
                    realValues = self.Agent.mainQ.predict(x=currStates)
                    for index in range(len(qValues)):
                        realValues[index][actions[index]] = qValues[index]
                    self.Agent.targetQ.fit(x=currStates, y=realValues, verbose=2)
                    self.Agent.updateTargetNetwork()

    def testing(self):
        done = True
        self.Env.reset()
        while done:
            _state, _parameter = self.Env.getState()
            done = _parameter[0]
            self.Env.setAction(self.numStringAction[self.Agent.getAction(_state, maxEpoch)])



if __name__=='__main__':
