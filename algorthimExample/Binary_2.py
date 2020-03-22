#https://programmers.co.kr/learn/courses/30/lessons/43238
n = 6
times = [7, 10]


def sumPeople(answer, times):
    _list = [answer // x for x in times]
    return sum(_list)


def solution(n, times):
    left, mid, right = min(times) * n // len(times), min(times) * n // len(times), max(times) * n

    while True:
        print(left, mid, right)
        if sumPeople(mid, times) < n:
            if sumPeople(mid + 1, times) >= n:
                # break
                return mid + 1
            else:
                left = mid
                mid = (mid + right) // 2
        else:
            right = mid
            mid = (mid + left) // 2
