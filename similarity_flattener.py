from abc import ABC, abstractmethod
from typing import List
import numpy as np


class SimilarityFlatteningStrategy(ABC):

    @abstractmethod
    def flatten_to_single_score(self, scores: List[float]) -> float:
        pass


class AverageStrategy(SimilarityFlatteningStrategy):

    def flatten_to_single_score(self, scores: List[float]) -> float:
        return np.mean(np.array(scores))


class MaxStrategy(SimilarityFlatteningStrategy):

    def flatten_to_single_score(self, scores: List[float]) -> float:
        return np.max(np.array(scores))
