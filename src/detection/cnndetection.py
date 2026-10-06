"""Adaptador para usar o modelo pré-treinado do submódulo CNNDetection"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from .base import Detector
from src.config import CNNDETECTION_DIR, WEIGHTS_PATH


class CNNDetectionDetector(Detector):
    """Detector de imagens sintéticas baseado no CNNDetection.

    O score retornado por :meth:`predict` é a probabilidade (entre 0 e 1) de
    a imagem ser sintética. O carregamento é adiado até a criação da instância
    para que importar o restante do pipeline não exija PyTorch imediatamente.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        crop_size: Optional[int] = None,
    ) -> None:
        self.weights_path = Path(weights_path or WEIGHTS_PATH).resolve()
        self.crop_size = crop_size

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Pesos do CNNDetection não encontrados: {self.weights_path}"
            )

        self._torch, self._transform, self._model = self._load_model(device)

    @property
    def name(self) -> str:
        return "CNNDetection (Blur+JPEG 0.5)"

    def predict(self, image_path: str) -> float:
        """Retorna a probabilidade de ``image_path`` ser uma imagem sintética."""
        image_file = Path(image_path)
        if not image_file.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image_file}")

        from PIL import Image

        with Image.open(image_file) as image:
            tensor = self._transform(image.convert("RGB")).unsqueeze(0)

        with self._torch.no_grad():
            probability = self._model(tensor.to(self.device)).sigmoid().item()
        return float(probability)

    def _load_model(self, requested_device: Optional[str]):
        try:
            import torch
            from torchvision import transforms
        except ImportError as error:
            raise ImportError(
                "CNNDetection requer PyTorch e torchvision. "
                "Instale as dependências do requirements.txt."
            ) from error

        source_dir = Path(CNNDETECTION_DIR).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Submódulo CNNDetection não encontrado: {source_dir}"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        from networks.resnet import resnet50

        self.device = torch.device(
            requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = resnet50(num_classes=1)
        try:
            checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        except TypeError:  # Compatibilidade com versões antigas do PyTorch.
            checkpoint = torch.load(self.weights_path, map_location="cpu")

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        transform_steps = []
        if self.crop_size is not None:
            transform_steps.append(transforms.CenterCrop(self.crop_size))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return torch, transforms.Compose(transform_steps), model
