n = 6
edge = [[3, 6], [4, 3], [3, 2], [1, 3], [1, 2], [2, 4], [5, 2]]
#https://programmers.co.kr/learn/courses/30/lessons/49189
def solution(n, edge):
    edgeDict = {i: [] for i in range(1, n + 1)}
    for [i, j] in edge:
        edgeDict[i].append(j)
        edgeDict[j].append(i)
    vistedNode, candidateNode = [1], [1]
    while True:
        nextNode = []
        for candidate in candidateNode:
            nextCandidate = [i for i in edgeDict[candidate]]
            nextNode.extend(nextCandidate)

        nextNode = set(nextNode).difference(set(vistedNode))
        candidateNode = list(nextNode)
        vistedNode += candidateNode
        if len(vistedNode) == n:
            break
    return len(candidateNode)
