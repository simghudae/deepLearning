def BFS(graph, startNode):
    #graph is dict type
    visit, queue = [], []
    queue.append(startNode)
    while queue:
        node = queue.pop(0)
        if node not in visit:
            visit.append(node)
            queue.append(graph[node])

    return visit