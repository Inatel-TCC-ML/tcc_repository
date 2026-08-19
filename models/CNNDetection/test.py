from pathlib import Path
import csv

from detection import CNNDetector


def score_frames(frames_dir, csv_path=None, threshold=0.5):
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Pasta de frames não encontrada: {frames_dir}"
        )

    # Pega somente imagens
    image_paths = sorted([
        path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not image_paths:
        raise ValueError(
            f"Nenhuma imagem encontrada em: {frames_dir}"
        )

    # Carrega o modelo UMA única vez
    detector = CNNDetector()

    results = []

    for image_path in image_paths:

        score = detector.predict(str(image_path))

        classification = "IA" if score > threshold else "REAL"

        result = {
            "frame": image_path.name,
            "score": score,
            "classification": classification
        }

        results.append(result)

        print(
            f"{image_path.name} | "
            f"score: {score:.6f} | "
            f"{classification}"
        )

    # Se quiser salvar os resultados
    if csv_path is not None:

        csv_path = Path(csv_path)

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "frame",
                    "score",
                    "classification"
                ]
            )

            writer.writeheader()
            writer.writerows(results)

        print(f"\nResultados salvos em: {csv_path}")

    return results