"""Gera métricas dos detectores e ensembles a partir do CSV do experimento.

Execute com::

    make metrics

O resultado é salvo em ``results/metrics_by_resolution.csv``. Cada linha
representa um detector individual ou uma estratégia de ensemble em uma
resolução específica.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


INPUT_CSV = Path("results/experiment_resolution.csv")
OUTPUT_CSV = Path("results/metrics_by_resolution.csv")
THRESHOLD = 0.5

# Há apenas uma imagem RAISE no CSV atual. Ela não é representativa da fonte,
# portanto fica fora das métricas até que exista uma amostra suficiente.
EXCLUDED_REAL_SOURCES = {"raise"}

DETECTOR_COLUMNS = (
    "UnivFD (CLIP ViT-L/14)",
    "CNNDetection (Blur+JPEG 0.5)",
    "NPR (ResNet50)",
    "B-Free (DINOv2)",
    "SPAI (Spectral AI-Generated Image Detector)",
)


def calculate_metrics(labels: list[int], scores: list[float]) -> dict[str, float | int]:
    """Calcula matriz de confusão e métricas para scores entre 0 e 1."""
    predictions = [int(score >= THRESHOLD) for score in scores]

    true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(labels) if labels else 0.0

    # AUC precisa das duas classes; o experimento balanceado sempre deve tê-las.
    auc_roc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")

    return {
        "Amostras": len(labels),
        "Fakes": sum(labels),
        "Reais": len(labels) - sum(labels),
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,
        "Acurácia": accuracy,
        "Precisão": precision,
        "Recall": recall,
        "Especificidade": specificity,
        "F1-score": f1_score,
        "AUC-ROC": auc_roc,
    }


def ensemble_scores(detector_scores: list[list[float]]) -> dict[str, list[float]]:
    """Calcula um score de ensemble para cada imagem e cada estratégia."""
    score_array = np.asarray(detector_scores, dtype=float)
    return {
        "Ensemble (média)": score_array.mean(axis=1).tolist(),
        "Ensemble (mediana)": np.median(score_array, axis=1).tolist(),
        "Ensemble (percentil 75)": np.percentile(score_array, 75, axis=1).tolist(),
        "Ensemble (percentil 90)": np.percentile(score_array, 90, axis=1).tolist(),
        "Ensemble (máximo)": score_array.max(axis=1).tolist(),
        "Ensemble (mínimo)": score_array.min(axis=1).tolist(),
        "Ensemble (votação majoritária)": (score_array >= THRESHOLD).mean(axis=1).tolist(),
    }


def keep_row(row: dict[str, str]) -> bool:
    """Remove a única imagem RAISE, que não forma uma fonte comparável."""
    return not (
        row["Tipo_Label"] == "Real"
        and row["Fonte_Real"].strip().lower() in EXCLUDED_REAL_SOURCES
    )


def load_rows() -> list[dict[str, str]]:
    """Lê e valida o arquivo de resultados da etapa de inferência."""
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"CSV de resultados não encontrado: {INPUT_CSV}")

    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if keep_row(row)]

    if not rows:
        raise ValueError("Nenhuma linha disponível para calcular as métricas.")

    missing_columns = set(DETECTOR_COLUMNS) - set(rows[0])
    if missing_columns:
        raise ValueError(
            "Colunas de detectores ausentes no CSV: "
            f"{', '.join(sorted(missing_columns))}"
        )
    return rows


def main() -> None:
    rows = load_rows()
    resolutions = sorted({row["Resolução"] for row in rows})

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Resolução",
                "Tipo",
                "Modelo",
                "Threshold",
                "Amostras",
                "Fakes",
                "Reais",
                "TP",
                "FP",
                "FN",
                "TN",
                "Acurácia",
                "Precisão",
                "Recall",
                "Especificidade",
                "F1-score",
                "AUC-ROC",
            ],
        )
        writer.writeheader()

        for resolution in resolutions:
            subset = [row for row in rows if row["Resolução"] == resolution]
            labels = [int(row["Tipo_Label"] == "Fake") for row in subset]

            detector_scores = [
                [float(row[column]) for column in DETECTOR_COLUMNS]
                for row in subset
            ]

            for detector_index, detector_name in enumerate(DETECTOR_COLUMNS):
                scores = [scores_by_image[detector_index] for scores_by_image in detector_scores]
                writer.writerow({
                    "Resolução": resolution,
                    "Tipo": "Detector individual",
                    "Modelo": detector_name,
                    "Threshold": THRESHOLD,
                    **calculate_metrics(labels, scores),
                })

            for ensemble_name, scores in ensemble_scores(detector_scores).items():
                writer.writerow({
                    "Resolução": resolution,
                    "Tipo": "Ensemble",
                    "Modelo": ensemble_name,
                    "Threshold": THRESHOLD,
                    **calculate_metrics(labels, scores),
                })

    print(f"Métricas salvas em: {OUTPUT_CSV}")
    print(f"Imagens por resolução: {len(rows) // len(resolutions)}")
    print(f"Resoluções analisadas: {', '.join(resolutions)}")


if __name__ == "__main__":
    main()
