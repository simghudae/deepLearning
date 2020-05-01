nums = [1,2,7,6,4]
#https://programmers.co.kr/learn/courses/30/lessons/12977

def solution(nums):
    primeList = primeNumber()
    result = 0
    candidateList = [[i, j, k] for i in range(len(nums)) for j in range(i + 1, len(nums)) for k in
                     range(j + 1, len(nums))]

    for candidate in candidateList:
        if nums[candidate[0]] + nums[candidate[1]] + nums[candidate[2]] in primeList:
            result += 1
    return result


def primeNumber():
    candidates = [i for i in range(2, 3000)]
    i = 0
    while True:
        candidates = [candidate for candidate in candidates if
                      candidate % candidates[i] != 0 or candidate <= candidates[i]]
        i += 1
        if i + 1 > len(candidates):
            break
    return candidates