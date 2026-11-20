from dataclasses import dataclass

from .agents import AgentResponse, MockAgent


@dataclass
class SelectedAgent:
    agent: MockAgent
    response: AgentResponse
    relevance: float


def node_sampling(agents, question, responses, k=3):
    scored = []
    for agent in agents:
        relevance = agent.relevance(question["domain"])
        score = relevance + 0.05 * agent.card.reliability
        scored.append((score, agent))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, agent in scored[:k]:
        response = responses[agent.card.name]
        selected.append(SelectedAgent(agent, response, score))
    return selected


def _order_selected(selected):
    ordered = sorted(
        selected,
        key=lambda x: (x.relevance, x.response.confidence),
        reverse=True,
    )
    return ordered


def message_passing(selected, rounds=2):
    ordered = _order_selected(selected)
    responses = [item.response for item in ordered]

    for _ in range(rounds):
        # Forward pass: higher-relevance agents influence lower-relevance agents.
        for i in range(1, len(responses)):
            upstream = responses[i - 1]
            current = responses[i]
            if upstream.confidence > current.confidence + 0.04:
                current.answer = upstream.answer
                current.confidence = min(0.98, upstream.confidence * 0.92)

        # Reverse pass: lower-relevance agents can send corrective evidence back.
        for i in range(len(responses) - 2, -1, -1):
            downstream = responses[i + 1]
            current = responses[i]
            if downstream.confidence > current.confidence:
                current.answer = downstream.answer
                current.confidence = min(0.98, downstream.confidence * 0.95)

    return responses


def graph_pooling(responses, mode="max"):
    if mode == "max":
        best = max(responses, key=lambda r: r.confidence)
        return best.answer

    # Weighted voting pooling.
    votes = {}
    for r in responses:
        votes[r.answer] = votes.get(r.answer, 0.0) + r.confidence
    return max(votes.items(), key=lambda item: item[1])[0]


def run_goa(agents, question, responses, k=3, pool_mode="max"):
    selected = node_sampling(agents, question, responses, k=k)
    updated = message_passing(selected)
    return graph_pooling(updated, mode=pool_mode)
