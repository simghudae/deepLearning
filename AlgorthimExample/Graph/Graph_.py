N, K = 5, 3
road = [[1, 2, 1], [2, 3, 3], [5, 2, 2], [1, 4, 2], [5, 3, 1], [5, 4, 2]]


def solution(N, road, K):
    pathRoad = {i: [] for i in range(1, N + 1)}
    for _road in road:
        pathRoad[_road[0]].append([_road[1], _road[2]])
        pathRoad[_road[1]].append([_road[0], _road[2]])

    resultList = [[] for _ in range(N)]
    while True:

        pathRoad[1]

