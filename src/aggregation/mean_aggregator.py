from .base import Aggregator


class MeanAggregator(Aggregator):
    """Combina scores de modelos usando média aritmética simples."""

    def aggregate(self, scores: dict[str, float]) -> float:
        if not scores:
            raise ValueError("É necessário ao menos um score para agregação.")
        return sum(scores.values()) / len(scores)
