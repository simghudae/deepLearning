arrows = [6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, 1, 6, 5, 5, 3, 6, 0]


# answer = 3

def solution(arrows):
    codeNumb = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
    vistedNode, Node = [[0, 0]], [0, 0]
    answer = 0
    for arrow in arrows:
        Node = [node + code for node, code in zip(Node, codeNumb[arrow])]
        if Node in vistedNode:
            if code == 0:
                answer += 1
                code = 1
        elif Node not in vistedNode:
            vistedNode.append(Node)
            code = 0
    return answer

# 이전시점이 방문하지 않고, 이후시점이 방문하지 않은 점인 경우 (그냥 진행)
# 이전시점이 방문했고, 이후시점도 방문한경우 (그냥 진행)
# 이전시점이 방문하지 않았고, 이후시점에 방문한 경우(도형완성)
# 이전시점에 방문했고, 이후시점에 방문하지 않은 경우(그냥 진행)
