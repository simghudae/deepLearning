skill_trees = ["BACDE", "CBADF", "AECB", "BDA"]
skill = "DE"

#https://programmers.co.kr/learn/courses/30/lessons/49993

def solution(skill, skill_trees):
    skill_trees = [[code for code in skillTree if code in skill] for skillTree in skill_trees]
    answer = 0
    for skillTree in skill_trees:
        if len(skillTree) == 0:
            answer += 1
        else:
            for i, code in enumerate(skillTree):
                if skill[i] == code:
                    if i + 1 == len(skillTree):
                        answer += 1
                else:
                    break
    return answer
