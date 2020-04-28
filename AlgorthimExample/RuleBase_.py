dirs = "ULURRDLLU"


def solution(dirs):
    keyDict = {"U": [0, 1], "D": [0, -1], "L": [-1, 0], "R": [1, 0]}
    keyDict2 = {"U": 0, "D": 1, "L": 2, "R": 3}
    left, right, answer = 0, 0, 0
    posDict = {str(i) + str(j): [0, 0, 0, 0] for i in range(-5, 6) for j in range(-5, 6)}
    for i, dir in enumerate(dirs):
        left, right = left + keyDict[dir][0], right + keyDict[dir][1]
        if str(left) + str(right) in posDict:
            if posDict[str(left) + str(right)][keyDict2[dir]] == 0:
                posDict[str(left) + str(right)][keyDict2[dir]] == 1

                answer += 1

    return answer


def solution(dirs):
    keyDict = {"U": [0, [0, 1]], "D": [1, [0, -1]], "L": [2, [-1, 0]], "R": [3, [1, 0]]}
    posDict = {str(i) + ',' + str(j): [0, 0, 0, 0] for i in range(-5, 6) for j in range(-5, 6)}
    left, right, answer = 0, 0, 0

    for dir in dirs:
        if str(left + keyDict[dir][1][0]) + ',' + str(right + keyDict[dir][1][1]) in posDict:
            if posDict[str(left) + ',' + str(right)][keyDict[dir][0]] == 0 or :
                posDict[str(left) + ',' + str(right)][keyDict[dir][0]] = 1
                answer += 1
            left, right = left + keyDict[dir][1][0], right + keyDict[dir][1][1]
    return answer
