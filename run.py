"""Analisa imagens do CommunityForensics-Eval com vários modelos."""

import csv
import gc
import os
from io import BytesIO
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image

from src.aggregation import (
    MaxScoreAggregator,
    MinScoreAggregator,
    MeanScoreAggregator,
    MedianScoreAggregator,
    PercentileScoreAggregator,
    MajorityVotingAggregator,
    AUCEvaluator
)
from src.detection import (
    BFreeDetector,
    CNNDetectionDetector,
    NPRDetector,
    SPAIDetector,
    UnivFDDetector,
)


# ---------- Configurações ---------- #

DATASET_NAME = "OwensLab/CommunityForensics-Eval"
DATASET_SPLIT = "CompEval"
CSV_FILENAME = "results.csv"

LIMIT_PER_ORIGIN = 100
MAX_IMAGE_PIXELS = 1_000_000
VOTING_THRESHOLD = 0.5


def create_detectors():
    """Inicializa os modelos utilizados na análise."""
    return [
        UnivFDDetector(),
        CNNDetectionDetector(),
        NPRDetector(),
        BFreeDetector(),
        SPAIDetector(),
    ]


def get_image_metadata(example):
    """Extrai o nome, o rótulo e a origem da imagem."""
    image_name = example.get("image_name", "imagem_desconhecida")
    is_fake = str(example["label"]) == "1"

    label_type = "Fake" if is_fake else "Real"
    origin = (
        example["model_name"]
        if is_fake
        else f"Real_{example['real_source']}"
    )

    clean_origin = "".join(
        char for char in origin
        if char.isalnum() or char in ("-", "_")
    ).strip()

    return image_name, label_type, origin, clean_origin


def load_image(image_data):
    """Decodifica os bytes e retorna a imagem em RGB."""
    with Image.open(BytesIO(image_data)) as source:
        source.load()

        if source.width * source.height > MAX_IMAGE_PIXELS:
            return None

        return source.convert("RGB")

def load_image_with_size(image_data, size):
    """Aceita apenas imagens com a resolução definida."""
    with Image.open(BytesIO(image_data)) as source:
        source.load()

        if source.size != size:
            return None

        return source.convert("RGB")


def predict_image(image_path, detectors):
    """Executa cada detector e retorna suas pontuações."""
    scores = []

    # A inferência não precisa calcular gradientes.
    with torch.no_grad():
        for detector in detectors:
            print(f"Executando {detector.name}...", flush=True)

            score = detector.predict(str(image_path))
            scores.append(score)

            print(
                f"{detector.name}: {score:.4f}",
                flush=True,
            )

    return scores


def calculate_aggregates(scores, label_type):
    """Calcula as métricas de agregação das pontuações."""
    return [
        MeanScoreAggregator().aggregate(scores),
        MedianScoreAggregator().aggregate(scores),
        PercentileScoreAggregator(75).aggregate(scores),
        PercentileScoreAggregator(90).aggregate(scores),
        MaxScoreAggregator().aggregate(scores),
        MinScoreAggregator().aggregate(scores),
        MajorityVotingAggregator(VOTING_THRESHOLD).aggregate(scores),
        # AUCEvaluator().evaluate(label_type, scores)
    ]


def create_csv_writer(file, detectors):
    """Cria o cabeçalho se o arquivo CSV estiver vazio."""
    writer = csv.writer(file)

    if os.path.getsize(CSV_FILENAME) == 0:
        writer.writerow([
            "Imagem",
            "Origem_Modelo",
            "Tipo_Label",
            *[detector.name for detector in detectors],
            "Média",
            "Mediana",
            "Percentil 75",
            "Percentil 90",
            "Máximo",
            "Mínimo",
            "Votação da Maioria",
            # "AUC-ROC"
        ])
        file.flush()

    return writer


def process_example(index, example, detectors, writer, file):
    """Processa uma amostra e registra seus resultados."""
    image_name, label_type, origin, clean_origin = (
        get_image_metadata(example)
    )

    print(f"\nAmostra {index}: {image_name}", flush=True)

    image = None
    temp_path = Path(f"temp_{clean_origin}_{index}.png")

    try:
        # image = load_image(example["image_data"])
        image = load_image_with_size(example["image_data"], (1024, 1024))

        if image is None:
            print(
                f"Imagem ignorada por resolução: {image_name}",
                flush=True,
            )
            return False

        image.save(temp_path, format="PNG")

        scores = predict_image(temp_path, detectors)
        aggregates = calculate_aggregates(scores, label_type)

        writer.writerow([
            image_name,
            origin,
            label_type,
            *scores,
            *aggregates,
        ])
        file.flush()

        print(f"Imagem registrada: {image_name}", flush=True)
        return True

    except Exception as error:
        print(
            f"Erro na imagem {image_name}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return False

    finally:
        if image is not None:
            image.close()

        if temp_path.exists():
            temp_path.unlink()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Coordena o processamento do dataset."""
    print("Inicializando detectores...", flush=True)
    detectors = create_detectors()

    file_exists = (
        os.path.exists(CSV_FILENAME)
        and os.path.getsize(CSV_FILENAME) > 0
    )

    counts_by_origin = {}

    with open(CSV_FILENAME, "a", newline="", encoding="utf-8",) as file:
        writer = (
            csv.writer(file)
            if file_exists
            else create_csv_writer(file, detectors)
        )

        dataset = load_dataset(
            DATASET_NAME,
            split=DATASET_SPLIT,
            streaming=True,
        )

        print("Processando dataset...", flush=True)

        for index, example in enumerate(dataset):
            _, _, _, clean_origin = get_image_metadata(example)

            count = counts_by_origin.get(clean_origin, 0)

            if count >= LIMIT_PER_ORIGIN:
                continue

            processed = process_example(
                index,
                example,
                detectors,
                writer,
                file,
            )

            if processed:
                counts_by_origin[clean_origin] = count + 1

                print(
                    f"Progresso: {clean_origin} — "
                    f"{counts_by_origin[clean_origin]}/"
                    f"{LIMIT_PER_ORIGIN}",
                    flush=True,
                )

    print("\nAnálise concluída. Resultados salvos em results.csv!")


if __name__ == "__main__":
    main()
