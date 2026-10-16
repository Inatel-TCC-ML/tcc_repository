"""Estratégias para resumir uma sequência de scores em um score final."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import fmean, median
from typing import Sequence

import numpy as np

from sklearn.metrics import roc_auc_score


class ScoreAggregator(ABC):
    """Converte uma sequência de scores em um único score final."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def aggregate(self, scores: Sequence[float]) -> float:
        pass

    @staticmethod
    def _validate(scores: Sequence[float]) -> None:
        if not scores:
            raise ValueError("É necessário ao menos um score.")


class MeanScoreAggregator(ScoreAggregator):
    @property
    def name(self) -> str:
        return "média"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return fmean(scores)


class MaxScoreAggregator(ScoreAggregator):
    @property
    def name(self) -> str:
        return "máximo"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return max(scores)


class MinScoreAggregator(ScoreAggregator):
    @property
    def name(self) -> str:
        return "mínimo"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return min(scores)


class MedianScoreAggregator(ScoreAggregator):
    @property
    def name(self) -> str:
        return "mediana"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return float(median(scores))


class PercentileScoreAggregator(ScoreAggregator):
    def __init__(self, percentile: float) -> None:
        if not 0 <= percentile <= 100:
            raise ValueError("O percentil deve estar entre 0 e 100.")
        self.percentile = percentile

    @property
    def name(self) -> str:
        return f"percentil {self.percentile:g}"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return float(np.percentile(scores, self.percentile))


class MajorityVotingAggregator(ScoreAggregator):
    def __init__(self, threshold: float) -> None:
            if not 0 <= threshold <= 100:
                raise ValueError("O limiar deve estar entre 0 e 1.")
            self.threshold = threshold

    @property
    def name(self) -> str:
        return f"votação majoritária"

    def aggregate(self, scores: Sequence[float]) -> float:
        scores = np.asarray(scores, dtype=float)
        votes = scores > self.threshold
        return votes.mean(axis=-1)


class AUCEvaluator:
    """Calcula a AUC-ROC a partir dos rótulos reais e dos scores previstos."""

    @property
    def name(self) -> str:
        return "AUC-ROC"

    def evaluate(
        self,
        y_true: str,
        y_scores: Sequence[float],
    ) -> float:

        if y_true == "Fake":
            y_true = np.full(len(y_scores), 1)
        else: 
            y_true = np.full(len(y_scores), 0)

        if len(y_true) != len(y_scores):
            raise ValueError(
                "Os rótulos reais e os scores devem ter o mesmo tamanho."
            )

        if len(y_true) == 0:
            raise ValueError("É necessário fornecer ao menos uma amostra.")

        return float(roc_auc_score(y_true, y_scores))
