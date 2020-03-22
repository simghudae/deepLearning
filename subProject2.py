triangle = [[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]]


def solution(triangle, answer=[], layer=0):
    if layer == 0:
        answer = triangle[0]
    elif layer == (len(triangle) - 1):
        return max(answer)
    _answer = triangle[layer + 1]
    for i in range(len(_answer)):
        if i == 0:
            _answer[i] += answer[0]
        elif i == len(_answer) - 1:
            _answer[i] += answer[-1]
        else:
            _answer[i] += max(answer[i-1], answer[i])
    print(_answer, layer)
    solution(triangle, _answer, layer + 1)
