from abc import ABC, abstractmethod
from typing import List
import numpy as np


class SimilarityFlatteningStrategy(ABC):

    def __init__(self):
        self.similarity_for_empty_list = 0.0

    @abstractmethod
    def flatten_to_single_score(self, scores: List[float]) -> float:
        pass


class AverageStrategy(SimilarityFlatteningStrategy):

    def flatten_to_single_score(self, scores: List[float]) -> float:
        if len(scores) == 0:
            return self.similarity_for_empty_list

        nonzeroes = [score for score in scores if score > 0.0]

        if len(nonzeroes) < 1:
            return 0.0

        return np.mean(np.array(nonzeroes))


class MaxStrategy(SimilarityFlatteningStrategy):

    def flatten_to_single_score(self, scores: List[float]) -> float:
        if len(scores) < 1:
            return self.similarity_for_empty_list

        return np.max(np.array(scores))
