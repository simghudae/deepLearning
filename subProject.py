n = 6
edge = [[3, 6], [4, 3], [3, 2], [1, 3], [1, 2], [2, 4], [5, 2]]

from collections import deque, defaultdict


def BFS(edgeDict):
    q = deque([[1, 0]])
    visited, answer = [], []
    while q:
        node, count = q.popleft()
        if node not in visited:
            visited.append(node)
            answer.append([node, count])
            nodeList = [[i, count + 1] for i in edgeDict[node]]
            q += nodeList
    return answer


def solution(n, edge):
    edgeDict = defaultdict(list)
    for u, v in edge:
        edgeDict[u].append(v)
        edgeDict[v].append(u)

    answerList = BFS(edgeDict)
    answer = [i for i in answerList if i[1] == max(answerList, key=lambda x: x[1])[1]]
    return len(answer)


# ----------------------------------------------------------------------------------------------

def solution(n, edge, vistedNode=[1], candidateNode=[1]):
    _candidateNode = [i[0] for i in edge if i[0] not in vistedNode and i[1] in candidateNode]
    _candidateNode += [i[1] for i in edge if i[0] in candidateNode and i[1] not in vistedNode]
    candidateNode = list(set(_candidateNode))
    vistedNode += candidateNode
    if len(vistedNode) == n:
        return len(candidateNode)

    sol = solution(n, edge, vistedNode, candidateNode)
    return sol
