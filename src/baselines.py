import random

from .agents import AgentResponse


def _weighted_vote(responses):
    votes = {}
    for response in responses:
        votes[response.answer] = votes.get(response.answer, 0.0) + response.confidence
    return max(votes.items(), key=lambda item: item[1])[0]


def single_best(agents, question, responses):
    best_agent = max(agents, key=lambda a: a.relevance(question["domain"]))
    response = responses[best_agent.card.name]
    return response.answer


def all_six_vote(agents, question, responses):
    return _weighted_vote([responses[agent.card.name] for agent in agents])


def random_three_vote(agents, question, responses, rng):
    chosen = rng.sample(agents, 3)
    return _weighted_vote([responses[agent.card.name] for agent in chosen])
