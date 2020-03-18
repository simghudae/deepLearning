distance = 25
rocks = [2, 14, 11, 21, 17]
n = 2


def solution(distance, rocks, n):
    rocks.sort()
    rocks.append(distance)

    left, right = 1, distance

    while True:
        mid = (left + right) // 2
        prev, count = 0, 0

        for rock in rocks:
            if rock - prev < mid:
                count += 1
            else:
                prev = rock

        if left == right-1:
            break
        if count > n:
            right = mid
        elif count <= n:
            left = mid

    return left