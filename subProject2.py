skill = "CBD"
skill_trees = ["BACDE", "CBADF", "AECB", "BDA"]


def solution(skill, skill_trees):
    answer = 0
    for skill_tree in skill_trees:
        candidate = [x for x in skill_tree if x in skill]
        if not candidate:
            answer += 1
        for i in range(len(candidate)):
            if skill[i] != candidate[i]:
                break
            elif i + 1 == len(candidate):
                answer += 1
    return answer


numbers = [3, 30, 34, 5, 9, 10]
numbers = [996, 901, 89,8, 9, 920, 99,97,91]


def solution(numbers):
    answer = ''
    number = [[x1 / pow(10, len(format(x1))), len(format(x1))] for x1 in numbers]
    number.sort(key=lambda x: (int(x[0]*10),-x[1],x[0]))
    for _ in range(len(numbers)):
        _digit, _len = number.pop()
        answer += format(int(_digit * pow(10, _len)))

    return answer


numbers = list(map(str, numbers))
numbers.sort(key=lambda x:(x[0], x[1%len(x)]))

numbers.sort(key=lambda x:x*3, reverse=True)