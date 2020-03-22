#https://programmers.co.kr/learn/courses/30/lessons/43164
tickets = [['ICN', 'SFO'], ['ICN', 'ATL'], ['SFO', 'ATL'], ['ATL', 'ICN'], ['ATL', 'SFO']]

def solution(tickets, answer=["ICN"]):
    candidates = [index for index, ticket in enumerate(tickets) if ticket[0] == answer[-1]]
    candidates.sort(key=lambda i: tickets[i][1])
    if not tickets: return answer
    elif not candidates: return None

    for candidate in candidates:
        sol = solution(tickets[:candidate] + tickets[candidate + 1:], answer + [tickets[candidate][1]])
        if sol: return sol
