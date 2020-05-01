people = [70, 50, 80, 50]
limit = 100
#https://programmers.co.kr/learn/courses/30/lessons/42885

def solution(people, limit):
    heavyPeople = [i for i in people if i > limit // 2]
    lightPeople = [i for i in people if i <= limit // 2]
    lightPass = []

    heavyPeople.sort()
    lightPeople.sort(reverse=True)

    answer, flags = len(heavyPeople), False
    for i in range(len(heavyPeople)):
        if flags:
            break
        while True:
            if len(lightPeople) == 0:
                flags = True
                break
            _light = lightPeople.pop(0)
            if limit >= heavyPeople[i] + _light:
                break
            else:
                lightPass.append(_light)

    if len(lightPass) % 2 == 0:
        answer += len(lightPass) // 2
    else:
        answer += len(lightPass) // 2 + 1

    return answer