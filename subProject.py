n = 4
costs = [[0, 1, 1], [0, 2, 2], [1, 2, 5], [1, 3, 1], [2, 3, 8]]


def solution(n, costs):
    costDict = {i: [] for i in range(n)}
    for i in costs:
        costDict[i[0]].append([i[1], i[2]])
        costDict[i[1]].append([i[0], i[2]])


    while True:


    answer = 0
    return answer
