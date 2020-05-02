left, right = [3, 2, 5]	,	[2, 4, 1]

def solution(left, right):
    answerList = [[-1 for _ in range(len(left)+1)] for _ in range(len(right)+1)]
    answerList[0][0] = 0
    for i in range(len(left)):
        for j in range(len(right)):
            if answerList[i][j] == -1:
                continue
            if left[i] > right[j] and answerList[i][j + 1] < answerList[i][j] + right[j]:
                answerList[i][j + 1] = answerList[i][j] + right[j]
            if answerList[i + 1][j] < answerList[i][j]:
                answerList[i + 1][j] = answerList[i][j]
            if answerList[i + 1][j + 1] < answerList[i][j]:
                answerList[i + 1][j + 1] = answerList[i][j]

    answerList = [max(i) for i in answerList]
    return max(answerList)