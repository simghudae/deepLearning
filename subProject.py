k = 3
number = "1231234"


def solution(number, k):
    answer, count = number[0], 0

    for i in range(1, len(number)):
        if int(answer[-1]) > int(number[i]):
            answer += number[i]

        else:
            for j in range(len(answer) - 1, -1, -1):
                if int(answer[j]) < int(number[i]):
                    count += 1
                elif int(answer[j]) >= int(number[i]):
                    break
                elif count >= k:
                    break
            # print(j, answer, number[i])
            if len(answer) == 1:
                answer = number[i]
            else:
                answer = answer[:j + 1] + number[i]
        if count >= k:
            break

        print(number[i], answer, count)

    return answer
