"""Adaptador para usar o modelo pré-treinado do submódulo NPR-DeepfakeDetection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Union

from .base import Detector
from src.config import NPRDEEPFAKE_DIR, NPRDEEPFAKE_WEIGHTS_PATH


class NPRDetector(Detector):
    """Detector baseado no NPR-ResNet50 e no checkpoint oficial do NPR."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(
            weights_path or NPRDEEPFAKE_WEIGHTS_PATH
        ).resolve()

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Pesos do NPR não encontrados: {self.weights_path}"
            )

        self._torch, self._transform, self._model = self._load_model(device)

    @property
    def name(self) -> str:
        return "NPR (ResNet50)"

    def predict(self, image_path: str) -> float:
        image_file = Path(image_path)

        if not image_file.is_file():
            raise FileNotFoundError(
                f"Imagem não encontrada: {image_file}"
            )

        from PIL import Image

        with Image.open(image_file) as image:
            tensor = self._transform(
                image.convert("RGB")
            ).unsqueeze(0)

        with self._torch.no_grad():
            output = self._model(tensor.to(self.device))
            score = output.sigmoid().item()

        return float(score)

    def _load_model(self, requested_device: Optional[str]):
        try:
            import torch
            from torchvision import transforms
        except ImportError as error:
            raise ImportError(
                "NPR requer PyTorch e torchvision. "
                "Instale requirements.txt."
            ) from error

        source_dir = Path(NPRDEEPFAKE_DIR).resolve()

        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Submódulo NPR não encontrado: {source_dir}"
            )

        # Carrega o resnet.py do NPR diretamente pelo caminho do arquivo.
        # Isso evita conflitos com outros submódulos que também possuem
        # um pacote chamado "networks".
        resnet_path = source_dir / "networks" / "resnet.py"

        if not resnet_path.is_file():
            raise FileNotFoundError(
                f"Arquivo resnet.py do NPR não encontrado: {resnet_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "npr_official_resnet",
            resnet_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Não foi possível carregar o módulo NPR: {resnet_path}"
            )

        npr_resnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(npr_resnet)

        self.device = torch.device(
            requested_device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = npr_resnet.resnet50(num_classes=1)

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                self.weights_path,
                map_location="cpu",
            )

        state_dict = checkpoint["model"]

        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return torch, transform, model
