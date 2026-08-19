"""Analisa a única imagem colocada em ``datasets/input`` com CNNDetection."""

from pathlib import Path

from src.config import INPUT_IMAGE_DIR
from src.detection import CNNDetectionDetector


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
    if len(images) > 1:
        names = "\n".join(f"- {image.name}" for image in images)
        raise RuntimeError(
            "Há mais de uma imagem em datasets/input. Deixe apenas uma para análise:\n"
            f"{names}"
        )
    return images[0]


def main() -> None:
    image_path = find_input_image(Path(INPUT_IMAGE_DIR))
    detector = CNNDetectionDetector()
    score = detector.predict(str(image_path))

    print(f"Imagem analisada: {image_path.name}")
    print(f"Score de imagem sintética: {score:.6f} ({score:.2%})")
    print("Classificação: possivelmente sintética" if score >= 0.5 else "Classificação: possivelmente real")


if __name__ == "__main__":
    main()
