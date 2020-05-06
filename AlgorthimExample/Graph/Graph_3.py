N, K = 5, 3
road = [[1, 2, 1], [2, 3, 3], [5, 2, 2], [1, 4, 2], [5, 3, 1], [5, 4, 2]]
# https://programmers.co.kr/learn/courses/30/lessons/12978

def solution(N, road, K):
    nodeDict = {i + 1: [] for i in range(N)}
    answerNode = [K + 1 for _ in range(N - 1)]
    for oneRoad in road:
        nodeDict[oneRoad[0]].append([oneRoad[1], oneRoad[2]])
        nodeDict[oneRoad[1]].append([oneRoad[0], oneRoad[2]])

    currNodes = [[1, 0]]

    while True:
        _currNodes = []
        for currNode in currNodes:
            _currNode = []
            for node in nodeDict[currNode[0]]:
                candidateNode = [node[0], currNode[1] + node[1]]
                if candidateNode[0] != 1 and candidateNode[1] < answerNode[candidateNode[0] - 2] and candidateNode not in _currNodes and candidateNode not in currNodes:
                    _currNodes.append(candidateNode)
                    answerNode[candidateNode[0] - 2] = candidateNode[1]
            _currNodes.extend(_currNode)
        if len(_currNodes) == 0:
            break
        currNodes.extend(_currNodes)
    return len([i for i in answerNode if i <= K]) + 1

