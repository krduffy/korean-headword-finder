from typing import List
from similarity_flattener import SimilarityFlatteningStrategy
import torch


class SimilarityCalculator:

    def __init__(
        self,
        known_usage_similarity_flattener: SimilarityFlatteningStrategy,
        sense_similarity_flattener: SimilarityFlatteningStrategy,
        definition_similarity_flattener: SimilarityFlatteningStrategy,
    ):
        self.known_usage_similarity_flattener = known_usage_similarity_flattener()
        self.sense_similarity_flattener = sense_similarity_flattener()
        self.definition_similarity_flattener = definition_similarity_flattener()

    def _get_similarity_of_tensors(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        t1, t2 = t1.squeeze(0), t2.squeeze(0)
        similarity = torch.cosine_similarity(t1, t2, dim=0)
        return similarity.item()

    def get_similarities_of_sense_definitions(
        self,
        unknown_usage_embedding: torch.Tensor,
        sense_definition_embeddings: List[torch.Tensor],
    ) -> List[float]:
        before_flattening = [
            [
                self._get_similarity_of_tensors(
                    unknown_usage_embedding, definition_embedding
                )
                for definition_embedding in sense_definition_set_embeddings
            ]
            for sense_definition_set_embeddings in sense_definition_embeddings
        ]

        return [
            self.definition_similarity_flattener.flatten_to_single_score(similarities)
            for similarities in before_flattening
        ]

    def get_similarities_of_sense_known_usages(
        self,
        unknown_usage_embedding: torch.Tensor,
        sense_known_usage_embeddings: List[List[List[torch.Tensor]]],
    ) -> List[float]:

        headword_similarities = []

        for headword in sense_known_usage_embeddings:
            sense_similarities = []

            for known_sense in headword:
                # Calculate similarities for all usage examples of this sense.
                usage_similarities = [
                    self._get_similarity_of_tensors(
                        unknown_usage_embedding, known_embedding
                    )
                    for known_embedding in known_sense
                ]

                # Flatten similarities for this sense.
                sense_similarity = (
                    self.known_usage_similarity_flattener.flatten_to_single_score(
                        usage_similarities
                    )
                )
                sense_similarities.append(sense_similarity)

            # Flatten sense similarities for this headword.
            headword_similarity = (
                self.sense_similarity_flattener.flatten_to_single_score(
                    sense_similarities
                )
            )
            headword_similarities.append(headword_similarity)

        return headword_similarities
