computers = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
n = 3

def dfs(graph, startNode, answerList):
    visit, stack = [], []
    stack.append(startNode)

    while stack:
        node = stack.pop()
        if node not in visit:
            visit.append(node)
            answerList[node] = 1
            _list = [i for i, x in enumerate(graph[node]) if x == 1]
            _list.remove(node)
            stack.extend(_list)


def solution(n, computers):
    answerList = [0 for _ in range(n)]
    answer = 0
    for i in range(len(computers)):
        if answerList[i] == 0:
            answer += 1
            dfs(computers, i, answerList)

        else:
            continue

    return answer
