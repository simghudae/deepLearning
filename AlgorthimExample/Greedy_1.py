name = 'JEROEN'
name = 'JAN'
#https://programmers.co.kr/learn/courses/30/lessons/42860
def solution(name):
    answerList = [min(ord(i) - ord('A'), ord('Z') + 1 - ord(i)) for i in name]
    cursor, answer = 0, 0
    while sum(answerList) != 0:
        left, right = 1, 1
        for _ in range(1, len(name)):
            _index = (cursor + right) % len(answerList)
            if answerList[_index] == 0:
                right += 1
            else:
                break
        for i in range(1, len(name)):
            _index = (cursor - left) %len(answerList)
            if answerList[_index] == 0:
                left += 1
            else:
                break
        answer += answerList[cursor] + 1
        answerList[cursor] = 0
        if left < right:
            cursor -= 1
        else:
            cursor += 1

    return answer-1
