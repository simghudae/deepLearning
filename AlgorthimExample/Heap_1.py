scoville = [1, 2, 3, 9, 10, 12]
K = 7

def solution(scoville, K):
    import heapq
    heapq.heapify(scoville)
    answer = 0
    while True:
        if min(scoville) > K:
            break
        elif len(scoville) < 2 or (K != 0 and max(scoville) == 0):
            return -1
        heapq.heappush(scoville, heapq.heappop(scoville) + 2 * heapq.heappop(scoville))
        answer += 1
    return answer

solution(scoville, K)
