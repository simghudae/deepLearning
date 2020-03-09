scoville = [1, 2, 3, 9, 10, 12]
K = 7


def solution(scoville, K):
    answer = 0
    while True:
        if min(scoville) > K:
            break
        elif len(scoville) < 2 or (K != 0 and max(scoville == 0)):
            answer = -1
            break

        first = scoville.pop(scoville.index(min(scoville)))
        second = scoville.pop(scoville.index(min(scoville)))
        scoville.append(first + 2 * second)
        answer += 1

    return answer
