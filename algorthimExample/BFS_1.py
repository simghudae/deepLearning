def BFS(graph, startNode):
    visit, queue = [], []
    queue.append(startNode)
    while queue:
        node = queue.pop(0)
        if node not in visit:
            visit.append(node)
            queue.append(graph[node])

    return visit


def diff(node, word):
    _list = [1 if _a != _b else 0 for _a, _b in zip(node, word)]
    return sum(_list)


def solution(begin, target, words):
    answer = 0
    visit, queue, _queue = [], [], []
    queue.append(begin)

    while queue:
        node = queue.pop(0)
        if node == target:
            return answer

        elif node not in visit:
            visit.append(node)
            _list = [_w for _w in words if diff(node, _w) == 1]
            _queue.extend(_list)

        if not queue:
            queue.extend(_queue)
            _queue = []
            answer += 1

    else:
        return 0


begin = 'hit'
target = 'cog'
words = ['hot', 'dot', 'dog', 'lot', 'log', 'cog']

solution(begin, target, words)
