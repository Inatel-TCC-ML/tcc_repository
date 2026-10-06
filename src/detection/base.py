from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def predict(self, image_path: str) -> float:
        """Retorna score de probabilidade de ser gerado por IA (0-1)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do modelo, usado para rastrear origem do score."""
        pass