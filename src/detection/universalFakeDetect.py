"""Adaptador para usar o modelo pré-treinado do submódulo UniversalFakeDetect"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import sys
from pathlib import Path
from typing import Optional, Union

from .base import Detector
from src.config import UNIVFD_DIR, UNIVFD_WEIGHTS_PATH


class UnivFDDetector(Detector):
    """Detector baseado em CLIP ViT-L/14 e no checkpoint oficial do UnivFD."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:

        # Define o caminho dos pesos
        self.weights_path = Path(
            weights_path or UNIVFD_WEIGHTS_PATH
        ).resolve()

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Pesos do UnivFD não encontrados: {self.weights_path}"
            )
        
        # Carrega o modelo, os pesos e o pré-processamento da imagem
        self._torch, self._transform, self._model = self._load_model(device)

    @property
    def name(self) -> str:
        return "UnivFD (CLIP ViT-L/14)"

    def predict(self, image_path: str) -> float:
        """Retorna o score de a imagem ser sintética, entre 0 e 1."""
        
        image_file = Path(image_path)
        
        if not image_file.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image_file}")

        from PIL import Image

        # Garante que a imagem esteja em RGB
        with Image.open(image_file) as image:
            tensor = self._transform(image.convert("RGB")).unsqueeze(0)
        
        # Diminuindo o uso da memória, desativando o cálculo de gradientes
        # sigmoid().item() converte a saída em um valor entre 0 e 1
        with self._torch.no_grad():
            score = self._model(tensor.to(self.device)).sigmoid().item()

        return float(score)


    def _load_model(self, requested_device: Optional[str]):
        try:
            import torch
            from torchvision import transforms
        except ImportError as error:
            raise ImportError(
                "UnivFD requer PyTorch e torchvision. Instale requirements.txt."
            ) from error

        # localiza o código do UnovFD dentro do submódulo
        source_dir = Path(UNIVFD_DIR).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Submódulo UnivFD não encontrado: {source_dir}")
        
        # Permite carregar os módulos diretamente dentro do submódulo
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        # Importa a arquitetura diretamente do submódulo.
        from models import get_model

        # Usa GPU caso esteja disponível. Se não, usa CPU
        self.device = torch.device(
            requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Cria a arquitetura utilizada pelo UnifFD
        model = get_model("CLIP:ViT-L/14")

        # Carrega os pesos pré-treinados
        try:
            classifier_weights = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            classifier_weights = torch.load(
                self.weights_path, 
                map_location="cpu"
            )
        
        # Aplica os pesos à arquitetura
        model.fc.load_state_dict(classifier_weights)
        model.to(self.device)
        model.eval()

        # Ajustes da imagem antes da detecção
        transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return torch, transform, model
