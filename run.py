"""Executa o experimento de detecção no CommunityForensics-Eval."""

import csv
import gc
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image

from src.aggregation import (
    MajorityVotingAggregator,
    MaxScoreAggregator,
    MeanScoreAggregator,
    MedianScoreAggregator,
    MinScoreAggregator,
    PercentileScoreAggregator,
)
from src.detection import (
    BFreeDetector,
    CNNDetectionDetector,
    NPRDetector,
    SPAIDetector,
    UnivFDDetector,
)


# Configuração do experimento.
DATASET_NAME = "OwensLab/CommunityForensics-Eval"
DATASET_SPLIT = "CompEval"
RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "experiment_resolution.csv"
TEMP_DIR = RESULTS_DIR / "temp"

# Os nomes precisam coincidir com a coluna ``model_name`` do dataset.
FAKE_MODELS = (
    "DFGAN",
    "GALIP",
    "Dalle2",
    "Dalle3",
    "FLUX-dev",
    "Imagen3",
    "kandinsky_2_2",
    "MidjourneyV6_1",
    "stable_cascade",
)
FAKE_IMAGES_PER_MODEL = 100

REAL_SOURCES = ("coco", "ffhq", "laion")
REAL_IMAGES_PER_SOURCE = 250

RESOLUTIONS: dict[str, tuple[int, int] | None] = {
    "original": None,
    "256x256": (256, 256),
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
}
PRIMARY_RESOLUTION = next(iter(RESOLUTIONS))

# Limite para evitar carregar imagens grandes demais na memória.
MAX_IMAGE_PIXELS = 20_000_000
VOTING_THRESHOLD = 0.5
RANDOM_SEED = 42


def create_detectors():
    return [
        UnivFDDetector(),
        CNNDetectionDetector(),
        NPRDetector(),
        BFreeDetector(),
        SPAIDetector(),
    ]


def get_image_metadata(example: dict[str, Any]) -> dict[str, str]:
    """Extrai metadados que serão gravados junto aos scores."""
    is_fake = int(example["label"]) == 1
    real_source = str(example.get("real_source") or "desconhecida")
    model_name = str(example.get("model_name") or "desconhecido")

    return {
        "image_name": str(example.get("image_name") or "imagem_desconhecida"),
        "label": "Fake" if is_fake else "Real",
        "origin": model_name if is_fake else f"Real_{real_source}",
        "model_name": model_name,
        "real_source": real_source,
        "architecture": str(example.get("architecture") or "desconhecida"),
    }


def load_image(image_data: bytes) -> Image.Image | None:
    with Image.open(BytesIO(image_data)) as source:
        source.load()
        if source.width * source.height > MAX_IMAGE_PIXELS:
            return None
        return source.convert("RGB")


def prepare_image(image: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    if size is None:
        return image.copy()
    return image.resize(size, Image.Resampling.LANCZOS)


def predict_image(image_path: Path, detectors) -> list[float]:
    scores = []
    with torch.no_grad():
        for detector in detectors:
            score = detector.predict(str(image_path))
            scores.append(float(score))
    return scores


def calculate_aggregates(scores: list[float]) -> list[float]:
    return [
        MeanScoreAggregator().aggregate(scores),
        MedianScoreAggregator().aggregate(scores),
        PercentileScoreAggregator(75).aggregate(scores),
        PercentileScoreAggregator(90).aggregate(scores),
        MaxScoreAggregator().aggregate(scores),
        MinScoreAggregator().aggregate(scores),
        MajorityVotingAggregator(VOTING_THRESHOLD).aggregate(scores),
    ]


def is_selected_example(metadata: dict[str, str], counts: dict[str, int]) -> str | None:
    """Retorna a chave da cota da amostra, ou None se ela não será usada."""
    if metadata["label"] == "Fake":
        model = metadata["model_name"]
        if model not in FAKE_MODELS or counts[model] >= FAKE_IMAGES_PER_MODEL:
            return None
        return model

    source = metadata["real_source"].lower()
    key = f"real:{source}"
    if source not in REAL_SOURCES or counts[key] >= REAL_IMAGES_PER_SOURCE:
        return None
    return key


def process_example(
    dataset_index: int,
    example: dict[str, Any],
    metadata: dict[str, str],
    detectors,
    writer: csv.writer,
    file,
) -> bool:
    image = load_image(example["image_data"])
    if image is None:
        print(f"Imagem ignorada por tamanho: {metadata['image_name']}", flush=True)
        return False

    try:
        for resolution_name, size in RESOLUTIONS.items():
            version = prepare_image(image, size)
            temp_path = TEMP_DIR / f"sample_{dataset_index}_{resolution_name}.png"

            try:
                # PNG evita introduzir compressão JPEG durante o redimensionamento.
                version.save(temp_path, format="PNG")
                scores = predict_image(temp_path, detectors)
                aggregates = calculate_aggregates(scores)

                writer.writerow([
                    dataset_index,
                    metadata["image_name"],
                    metadata["origin"],
                    metadata["label"],
                    metadata["model_name"],
                    metadata["real_source"],
                    metadata["architecture"],
                    resolution_name,
                    *scores,
                    *aggregates,
                ])
                file.flush()
                print(
                    f"  {resolution_name}: registrado ({metadata['image_name']})",
                    flush=True,
                )
            finally:
                version.close()
                if temp_path.exists():
                    temp_path.unlink()

        return True
    except Exception as error:
        print(
            f"Erro na imagem {metadata['image_name']}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return False
    finally:
        image.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_csv_writer(file, detectors) -> csv.writer:
    writer = csv.writer(file)
    writer.writerow([
        "Indice_Dataset",
        "Imagem",
        "Origem_Modelo",
        "Tipo_Label",
        "Modelo_Gerador",
        "Fonte_Real",
        "Arquitetura",
        "Resolução",
        *[detector.name for detector in detectors],
        "Média",
        "Mediana",
        "Percentil 75",
        "Percentil 90",
        "Máximo",
        "Mínimo",
        "Votação da Maioria",
    ])
    file.flush()
    return writer


def quotas_complete(counts: dict[str, int]) -> bool:
    return (
        all(counts[model] >= FAKE_IMAGES_PER_MODEL for model in FAKE_MODELS)
        and all(
            counts[f"real:{source}"] >= REAL_IMAGES_PER_SOURCE
            for source in REAL_SOURCES
        )
    )


def read_completed_counts(counts: dict[str, int]) -> dict[str, int]:
    """Recupera as cotas já concluídas ao reiniciar um experimento interrompido."""
    if not CSV_PATH.is_file() or CSV_PATH.stat().st_size == 0:
        return counts

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            # Conta cada imagem uma vez, usando a primeira resolução configurada.
            if row["Resolução"] != PRIMARY_RESOLUTION:
                continue

            if row["Tipo_Label"] == "Fake":
                model = row["Modelo_Gerador"]
                if model in counts:
                    counts[model] += 1
            else:
                source = row["Fonte_Real"].lower()
                key = f"real:{source}"
                if key in counts:
                    counts[key] += 1

    return counts


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    counts = {
        **{model: 0 for model in FAKE_MODELS},
        **{f"real:{source}": 0 for source in REAL_SOURCES},
    }
    counts = read_completed_counts(counts)

    if quotas_complete(counts):
        print(f"O experimento já está completo: {CSV_PATH}")
        return

    print("Inicializando detectores...", flush=True)
    detectors = create_detectors()

    if CSV_PATH.is_file() and CSV_PATH.stat().st_size > 0:
        open_mode = "a"
        print("Continuando o CSV existente...", flush=True)
    else:
        open_mode = "w"

    # Streaming evita carregar o dataset inteiro na memória.
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    dataset = dataset.shuffle(seed=RANDOM_SEED, buffer_size=100)

    with CSV_PATH.open(open_mode, newline="", encoding="utf-8-sig") as file:
        writer = (
            csv.writer(file)
            if open_mode == "a"
            else create_csv_writer(file, detectors)
        )
        print("Processando a amostra balanceada...", flush=True)

        for dataset_index, example in enumerate(dataset):
            metadata = get_image_metadata(example)
            quota_key = is_selected_example(metadata, counts)
            if quota_key is None:
                continue

            print(
                f"\nAmostra {dataset_index}: {metadata['origin']} "
                f"({metadata['label']})",
                flush=True,
            )
            if process_example(
                dataset_index, example, metadata, detectors, writer, file
            ):
                counts[quota_key] += 1
                print(
                    f"Progresso {quota_key}: {counts[quota_key]}/"
                    f"{FAKE_IMAGES_PER_MODEL if metadata['label'] == 'Fake' else REAL_IMAGES_PER_SOURCE}",
                    flush=True,
                )

            if quotas_complete(counts):
                break

    print("\nExperimento concluído.")
    print(f"Resultados: {CSV_PATH}")
    print("\nAmostras selecionadas:")
    for key, count in counts.items():
        print(f"  {key}: {count}")

    if not quotas_complete(counts):
        print(
            "\nAviso: alguma cota não foi preenchida. Confira os nomes "
            "em FAKE_MODELS e REAL_SOURCES.",
        )


if __name__ == "__main__":
    main()
