"""Adaptador para usar o detector SPAI oficial."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import Detector
from src.config import SPAI_DIR, SPAI_WEIGHTS_PATH


class SPAIDetector(Detector):
    """Detector de imagens sintéticas baseado no SPAI (CVPR 2025)."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(weights_path or SPAI_WEIGHTS_PATH).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Pesos do SPAI não encontrados: {self.weights_path}. "
                "Baixe o checkpoint oficial e salve-o como weights/spai.pth."
            )

        self.source_dir = Path(SPAI_DIR).resolve()
        if not (self.source_dir / "spai" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"Submódulo SPAI não encontrado ou incompleto: {self.source_dir}"
            )
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))

        try:
            torch = importlib.import_module("torch")
            image_module = importlib.import_module("PIL.Image")
            spai_config = importlib.import_module("spai.config")
            spai_models = importlib.import_module("spai.models")
            spai_utils = importlib.import_module("spai.utils")
        except ImportError as error:
            raise ImportError(
                "SPAI requer suas dependências de inferência. "
                "Instale os pacotes listados em requirements.txt."
            ) from error

        config_path = self.source_dir / "configs" / "spai.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuração oficial do SPAI não encontrada: {config_path}"
            )

        self._torch = torch
        self._image_module = image_module
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        config = spai_config.get_config({
            "cfg": str(config_path),
            "pretrained": str(self.weights_path),
            "opts": [],
        })
        model = spai_models.build_cls_model(config)
        spai_utils.load_pretrained(
            config,
            model,
            logging.getLogger(self.__class__.__name__),
            checkpoint_path=self.weights_path,
            verbose=False,
        )
        self._model = model.to(self.device)
        self._model.eval()
        self._image_size = config.DATA.IMG_SIZE
        self._feature_extraction_batch = config.MODEL.FEATURE_EXTRACTION_BATCH

    @property
    def name(self) -> str:
        return "SPAI (Spectral AI-Generated Image Detector)"

    def predict(self, image_path: str) -> float:
        image_file = Path(image_path)
        if not image_file.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image_file}")

        with self._image_module.open(image_file) as image:
            image = image.convert("RGB")
            image_tensor = self._torch.from_numpy(
                np.array(image, copy=True)
            ).permute(2, 0, 1).float().div_(255)

        height, width = image_tensor.shape[-2:]
        pad_left = max((self._image_size - width) // 2, 0)
        pad_right = max(
            (self._image_size - width + 1) // 2 if width < self._image_size else 0,
            0,
        )
        pad_top = max((self._image_size - height) // 2, 0)
        pad_bottom = max(
            (self._image_size - height + 1) // 2 if height < self._image_size else 0,
            0,
        )
        image_tensor = self._torch.nn.functional.pad(
            image_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
        ).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            logit = self._model(
                [image_tensor],
                self._feature_extraction_batch,
            )
            score = logit.sigmoid().item()

        return float(score)
