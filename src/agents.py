from dataclasses import dataclass
import random


@dataclass
class ModelCard:
    name: str
    specialty: str
    reliability: float
    cross_domain: float = 0.40


@dataclass
class AgentResponse:
    agent: str
    answer: int
    confidence: float


class MockAgent:
    def __init__(self, card: ModelCard):
        self.card = card

    def relevance(self, domain: str) -> float:
        if self.card.specialty == domain:
            return 1.0
        if self.card.specialty == "general":
            return 0.62
        if self.card.specialty == "reasoning":
            return 0.70
        if self.card.specialty == "none":
            return 0.15
        return self.card.cross_domain

    def answer(self, question, rng: random.Random) -> AgentResponse:
        domain = question["domain"]
        p_correct = self._correct_probability(domain)
        p_correct = min(0.92, max(0.10, p_correct))

        correct_idx = question["answer"]
        if rng.random() < p_correct:
            answer = correct_idx
        else:
            wrong = [i for i in range(len(question["options"])) if i != correct_idx]
            answer = rng.choice(wrong)

        confidence = p_correct + rng.uniform(-0.08, 0.08)
        confidence = min(0.98, max(0.05, confidence))
        return AgentResponse(self.card.name, answer, confidence)

    def _correct_probability(self, domain: str) -> float:
        if self.card.specialty == domain:
            return self.card.reliability + 0.12
        if self.card.specialty == "general":
            return 0.56
        if self.card.specialty == "reasoning":
            return 0.58 if domain != "medical" else 0.42
        if self.card.specialty == "none":
            return 0.22
        return self.card.cross_domain


def build_agents():
    cards = [
        ModelCard("MathAgent", "math", 0.82),
        ModelCard("CodeAgent", "coding", 0.82),
        ModelCard("MedAgent", "medical", 0.82),
        ModelCard("GeneralAgent", "general", 0.58),
        ModelCard("ReasonAgent", "reasoning", 0.66),
        ModelCard("NoisyAgent", "none", 0.20),
    ]
    return [MockAgent(card) for card in cards]
