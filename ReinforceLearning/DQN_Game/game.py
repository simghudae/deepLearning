import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random


class Game:
    def __init__(self, screenHeight, screenWidth, levelUp, controlUser=True, showGame=True):
        # set hyperparameter
        self.screenHeight, self.screenWidth = screenHeight, screenWidth
        self.levelUp, self.showGame, self.controlUser = levelUp, showGame, controlUser

        # setting model
        self.screen, self.patchList, self.gameNum, self.state = np.zeros([self.screenWidth, self.screenHeight]), [], 1, True
        self.reset()

        # show graphic
        if showGame:
            self.fig, self.axis = self._prepareDisplay()
            self._drawScreen()

        # continue games
        while self.state:
            self.oneGameUser()

    def oneGameUser(self):
        while self.gameState:
            if self.controlUser:
                action = str(input("please action insert : "))
            else:
                action = self.getAction()

            if action == 'stop':
                self.state, self.gameState = False, False
            self.oneStep(action)
        self.gameNum += 1
        self.reset()

    def getAction(self, action):
        return str(action)

    def oneStep(self, action):  # 게임의 진행
        self.reward += self._updateAll(action)
        self.step += 1
        if self.showGame:
            self._drawScreen()

    def _prepareDisplay(self):  # 게임의 화면을 보여주기 위한 것
        fig, axis = plt.subplots(figsize=(10, 6))
        plt.axis((0, self.screenWidth, 0, self.screenHeight))
        plt.draw()
        plt.ion()
        plt.show()
        return fig, axis

    def _drawScreen(self):
        for patch in self.patchList:
            patch.remove()

        _patchList = []
        self.axis.set_title("Game : {0}, Step {1}, Reward : {2}".format(self.gameNum, self.step, self.reward), fontsize=12)

        carState, boxState = self._getState()
        _patchList = [patches.Circle(xy=(box[0], box[1]), radius=0.5, facecolor='blue') for box in boxState]
        _patchList.extend(patches.Circle(xy=(box[0], box[1]), radius=0.1, facecolor='red') for box in boxState)
        _patchList.append(patches.Rectangle(xy=(carState[0] - 1 / 2, carState[1]), width=1, height=1, facecolor='blue'))
        _patchList.append(patches.Rectangle(xy=(carState[0], carState[1] + 1 / 2), width=0.1, height=0.1, facecolor='red'))

        for patch in _patchList:
            self.axis.add_patch(patch)

        self.fig.canvas.draw()
        self.patchList = _patchList

    def _getState(self):  # 게임의 상태를 가지고 오는것으로 2차원 배열에 사물이 있으면 1 없으면 0
        carWidth, carHeight = np.where(self.screen == 2)
        boxWidth, boxHeight = np.where(self.screen == 1)
        return [carWidth, carHeight], [[width, height] for width, height in zip(list(boxWidth), list(boxHeight))]

    def reset(self):  # 자동차와 장매울의 위치와 보상값을 초기화
        self.reward, self.screen[:], self.step = 0, 0, 0
        self.gameState = True
        self.screen[self.screen.shape[0] // 2, 0] = 2  # reset Car

    def _updateAll(self, move):
        carPosition, boxPosition = self._getState()
        carWidth, carHeight = carPosition

        # update Box
        countReward = 0
        for box in boxPosition:
            boxWidth, boxHeight = box
            if boxHeight == 0:
                self.screen[boxWidth, boxHeight] = 0
                countReward += 1

            elif boxHeight == 1 and self.screen[boxWidth, 0] == 2:
                continue

            elif boxHeight != 0:
                self.screen[boxWidth, boxHeight] = 0
                self.screen[boxWidth, boxHeight - 1] = 1

        # create new box
        boxNum = min(np.random.poisson(lam=self.step // self.levelUp), self.screenWidth - 1)
        boxList = [i for i in range(self.screenWidth)]
        random.shuffle(boxList)
        for _ in range(boxNum):
            width = boxList.pop()
            self.screen[width, self.screenHeight - 1] = 1

        # update car
        if move == 'left' and carWidth != 0:
            if self.screen[carWidth - 1, carHeight] == 1:
                self.gameState = False
            self.screen[carWidth, carHeight] = 0
            self.screen[carWidth - 1, carHeight] = 2
        elif move == "right" and carWidth + 1 != self.screenWidth:
            if self.screen[carWidth + 1, carHeight] == 1:
                self.gameState = False
            self.screen[carWidth, carHeight] = 0
            self.screen[carWidth + 1, carHeight] = 2
        return countReward


if __name__ == '__main__':
    aa = Game(5, 10, 5)