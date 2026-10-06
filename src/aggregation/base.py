from abc import ABC, abstractmethod

class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, scores: dict[str, float]) -> float:
        """Recebe {nome_modelo: score} e retorna score final agregado."""
        pass