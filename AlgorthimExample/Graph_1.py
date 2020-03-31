n = 5
results = [[4, 3], [4, 2], [3, 2], [1, 2], [2, 5]]

#https://programmers.co.kr/learn/courses/30/lessons/49191
def solution(n, results):
    answer = 0
    for _n in range(1, n + 1):
        winList = [i[0] for i in results if i[1] == _n]
        lossList = [i[1] for i in results if i[0] == _n]
        winList, lossList = set(winList), set(lossList)
        while True:
            _List = set([i[0] for i in results if i[1] in winList])
            if _List.issubset(winList):
                break
            winList = winList.union(_List)
        while True:
            _List = set([i[1] for i in results if i[0] in lossList])
            if _List.issubset(lossList):
                break
            lossList = lossList.union(_List)
        if len(winList) + len(lossList) == n - 1:
            answer += 1
    return answer
