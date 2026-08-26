"""Estratégias para resumir os scores dos keyframes de um vídeo."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import fmean, median
from typing import Sequence

import numpy as np


class FrameAggregator(ABC):
    """Converte uma sequência de scores de keyframes em um score de vídeo."""

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
            raise ValueError("É necessário ao menos um score de keyframe.")


class MeanFrameAggregator(FrameAggregator):
    @property
    def name(self) -> str:
        return "média"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return fmean(scores)


class MaxFrameAggregator(FrameAggregator):
    @property
    def name(self) -> str:
        return "máximo"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return max(scores)


class MedianFrameAggregator(FrameAggregator):
    @property
    def name(self) -> str:
        return "mediana"

    def aggregate(self, scores: Sequence[float]) -> float:
        self._validate(scores)
        return float(median(scores))


class PercentileFrameAggregator(FrameAggregator):
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
