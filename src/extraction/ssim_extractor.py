"""Extração de keyframes pela diferença estrutural (MSSIM) entre frames."""

from __future__ import annotations

from pathlib import Path

import cv2
from skimage.metrics import structural_similarity

from src.config import OUTPUT_DIR
from .base import KeyframeExtractor


class MSSIMKeyframeExtractor(KeyframeExtractor):
    """Salva um frame quando sua similaridade com o último keyframe é baixa."""

    def __init__(
        self,
        threshold: float = 0.90,
        comparison_width: int = 320,
        output_dir: str | Path | None = None,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold deve estar entre 0 e 1.")
        self.threshold = threshold
        self.comparison_width = comparison_width
        self.output_dir = Path(output_dir or OUTPUT_DIR)

    def extract(self, video_path: str) -> list[str]:
        video_file = Path(video_path)
        if not video_file.is_file():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_file}")

        target_dir = self.output_dir / "keyframes" / video_file.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_file}")

        keyframes: list[str] = []
        last_keyframe_gray = None
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                current_gray = self._prepare_for_comparison(frame)
                is_keyframe = (
                    last_keyframe_gray is None
                    or structural_similarity(last_keyframe_gray, current_gray)
                    < self.threshold
                )
                if is_keyframe:
                    frame_file = target_dir / f"frame_{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(frame_file), frame):
                        raise OSError(f"Não foi possível salvar o keyframe: {frame_file}")
                    keyframes.append(str(frame_file))
                    last_keyframe_gray = current_gray
                frame_index += 1
        finally:
            capture.release()

        if not keyframes:
            raise ValueError(f"O vídeo não possui frames legíveis: {video_file}")
        return keyframes

    def _prepare_for_comparison(self, frame):
        height, width = frame.shape[:2]
        if width > self.comparison_width:
            scaled_height = max(1, round(height * self.comparison_width / width))
            frame = cv2.resize(frame, (self.comparison_width, scaled_height))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
