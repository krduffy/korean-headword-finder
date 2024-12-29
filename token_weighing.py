from abc import ABC, abstractmethod
from typing import Set
from torch import Tensor, float32, ones


class TokenWeighingStrategy(ABC):
    def __init__(self, stopwords: Set[str]):
        self.stopwords = stopwords

    @abstractmethod
    def get_mask(self, tokens: Tensor) -> Tensor:
        pass


class DoNoWeighingStrategy(TokenWeighingStrategy):
    def get_mask(self, tokens: Tensor) -> Tensor:
        return ones(len(tokens), dtype=float32)


class StopWordsLowWeightStrategy(TokenWeighingStrategy):

    def _is_stopword(self, token: str) -> bool:
        if token in self.stopwords:
            return True

        if token.startswith("##") and token[2:] in self.stopwords:
            return True

        return False

    def get_mask(self, tokens: Tensor) -> Tensor:

        mask = ones(len(tokens), dtype=float32)

        for i, token in enumerate(tokens):
            # is a special token?
            # [UNK] is often for hanja in korean inputs
            if token in ["[CLS]", "[SEP]", "[UNK]", "[PAD]"]:
                mask[i] = 0.1

            # is punctuation ?
            if token in ["(", ")", ".", ","]:
                mask[i] = 0

            # is stop word ?
            if self._is_stopword(token):
                mask[i] = 0

        return mask
