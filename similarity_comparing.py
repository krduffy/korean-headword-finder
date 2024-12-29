from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn.functional as F


class SimilarityComparisonStrategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_similarity(self, t1: Tensor, t2: Tensor) -> float:
        """Returns the similarity of the vectors"""
        pass


class CosineSimilarityStrategy(SimilarityComparisonStrategy):

    def get_similarity(self, t1, t2):
        return F.cosine_similarity(t1, t2).item()


class EuclideanDistanceStrategy(SimilarityComparisonStrategy):

    def get_similarity(self, t1, t2):
        d = torch.linalg.norm(t1 - t2)
        return 1 / (d + 1)
