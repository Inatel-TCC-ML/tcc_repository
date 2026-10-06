"""Analisa imagems adicionadas em ``datasets/input`` com vários modelos"""

from pathlib import Path

import csv

from src.aggregation import (
    MaxScoreAggregator,
    MeanScoreAggregator,
    MedianScoreAggregator,
    PercentileScoreAggregator,
)

from src.config import INPUT_IMAGE_DIR
from src.detection import UnivFDDetector, CNNDetectionDetector

SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

def find_input_image(input_dir: Path) -> Path:
    """Localiza a única imagem de entrada e explica erros comuns ao usuário."""
    input_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em: {input_dir}\n"
            f"Coloque uma imagem ({extensions}) nessa pasta e execute novamente."
        )
    return images


def main() -> None:
    images = find_input_image(Path(INPUT_IMAGE_DIR))
    detectors = [
        UnivFDDetector(), 
        CNNDetectionDetector()
    ]

    with open("results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        models_name = [detector.name for detector in detectors]

        writer.writerow([
            "Imagem",
            *models_name,
            "Média",
            "Mediana",
            "Percentil 75",
            "Percentil 90",
            "Máximo",
        ])

        for image in images:
            scores = [detector.predict(str(image)) for detector in detectors]

            writer.writerow([
                image.name,
                *scores,
                MeanScoreAggregator().aggregate(scores),
                MedianScoreAggregator().aggregate(scores),
                PercentileScoreAggregator(75).aggregate(scores),
                PercentileScoreAggregator(90).aggregate(scores),
                MaxScoreAggregator().aggregate(scores),
            ])


if __name__ == "__main__":
    main()
