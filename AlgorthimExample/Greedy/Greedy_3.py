number = '10000'
k = 2

#https://programmers.co.kr/learn/courses/30/lessons/42883

def smallCase(n, j, answer, count, k):
    if len(answer) == 0:
        answer.append(n)
        return answer, count, j, False
    elif count >= k:
        answer.append(n)
        return answer, count, j, False
    elif answer[j] > n:
        answer.append(n)
        return answer, count, j, False
    elif answer[j] == n:
        answer.append(n)
        return answer, count, j, False
    elif answer[j] < n:
        answer.pop(j)
        return answer, count + 1, j - 1, True


def solution(number, k):
    ans, count = [int(number[0])], 0
    for i in range(1, len(number)):
        # print(ans, number[i], count)
        j, Flags = len(ans) - 1, True
        while Flags:
            ans, count, j, Flags = smallCase(int(number[i]), j, ans, count, k)

    answer = ''
    if count == k:
        for c in ans:
            answer += str(c)
    else:
        for i, c in enumerate(ans):
            if i < len(ans)-k:
                answer += str(c)
            else:
                break

    return answer
