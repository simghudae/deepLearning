#https://programmers.co.kr/learn/courses/30/lessons/43237
budgets = [120, 110, 140, 150]
M = 485


def calculationM(budgets, k):
    return sum([x if x < k else k for x in budgets])

def solution(budgets, M):
    budgets.sort()
    _left, _answer, _right = M // len(budgets), M // len(budgets), budgets[-1]
    if M//len(budgets)>budgets[-1]:
        return budgets[-1]

    while True:
        if calculationM(budgets, _answer) < M:
            if calculationM(budgets, _answer + 1) > M:
                break
            _left = _answer
            _answer = (_right + _answer) // 2

        elif calculationM(budgets, _answer) > M:
            _right = _answer
            _answer = (_left + _answer) // 2

        else:
            break
    return _answer


solution(budgets, M)
