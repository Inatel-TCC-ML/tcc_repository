from abc import ABC, abstractmethod

class KeyframeExtractor(ABC):
    @abstractmethod
    def extract(self, video_path: str) -> list[str]:
        """Retorna lista de caminhos das imagens extraídas."""
        pass