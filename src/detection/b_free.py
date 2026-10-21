"""Adaptador para usar o modelo pré-treinado do submódulo B-Free."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Union

from .base import Detector
from src.config import BFREE_DIR, BFREE_WEIGHTS_DIR


class BFreeDetector(Detector):
    """Detector baseado no B-Free (ViT DINOv2 com registers) e nos pesos oficiais."""

    def __init__(
        self,
        weights_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.weights_dir = Path(weights_dir or BFREE_WEIGHTS_DIR).resolve()
        self.config_path = self.weights_dir / "config.yaml"

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"config.yaml do B-Free não encontrado: {self.config_path}"
            )

        self._torch, self._transform, self._model = self._load_model(device)

    @property
    def name(self) -> str:
        return "B-Free (DINOv2)"

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

        # O modelo oficial recebe a imagem inteira e, internamente, calcula a
        # média dos logits de 5 recortes de 504x504 (centro e quatro cantos).
        with self._torch.no_grad():
            output = self._model(tensor.to(self.device))
            score = output[:, 0].sigmoid().item()

        return float(score)

    @staticmethod
    def _load_module(name: str, path: Path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Arquivo do B-Free não encontrado: {path}"
            )

        spec = importlib.util.spec_from_file_location(name, path)

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Não foi possível carregar o módulo B-Free: {path}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_model(self, requested_device: Optional[str]):
        try:
            import timm
            import torch
            import yaml
            from torchvision import transforms
        except ImportError as error:
            raise ImportError(
                "B-Free requer PyTorch, torchvision, timm e PyYAML. "
                "Instale requirements.txt."
            ) from error

        source_dir = Path(BFREE_DIR).resolve()

        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Submódulo B-Free não encontrado: {source_dir}"
            )

        # Carrega os arquivos do B-Free diretamente pelo caminho. Isso evita
        # conflitos com outros submódulos que também possuem os pacotes
        # "networks" e "utils".
        wrapper5crops = self._load_module(
            "bfree_official_wrapper5crops",
            source_dir / "networks" / "wrapper5crops.py",
        )
        normalization = self._load_module(
            "bfree_official_normalization",
            source_dir / "utils" / "normalization.py",
        )

        with open(self.config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        arch = config["arch"]
        weights_path = self.weights_dir / config["weights_file"]

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Pesos do B-Free não encontrados: {weights_path}"
            )

        # Equivale a networks.get_network(arch) do repositório oficial.
        prefix = "timm_c5i504_"

        if not arch.startswith(prefix):
            raise ValueError(f"Arquitetura do B-Free não suportada: {arch}")

        self.device = torch.device(
            requested_device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        backbone = timm.create_model(
            arch[len(prefix):],
            num_classes=1,
            pretrained=False,
        )
        model = wrapper5crops.Wrapper5crops(backbone, 504)

        try:
            checkpoint = torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                weights_path,
                map_location="cpu",
            )

        model.load_state_dict(checkpoint["model"])

        model.to(self.device)
        model.eval()

        transform = transforms.Compose(
            normalization.get_list_norm(config["norm_type"])
        )

        return torch, transform, model
