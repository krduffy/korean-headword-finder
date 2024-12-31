from typing import List, Tuple
from embedder import Embedder
from similarity_calculator import SimilarityCalculator
from similarity_flattener import SimilarityFlatteningStrategy
from test_types import KnownHeadwordInformation, Language


class MatchingUsageHeadwordDisambiguator:

    def __init__(
        self,
        language: Language,
        definition_weight: float,
        known_usage_similarity_flattener: SimilarityFlatteningStrategy,
        sense_similarity_flattener: SimilarityFlatteningStrategy,
    ):
        self.definition_weight = definition_weight
        self.similarity_calculator = SimilarityCalculator(
            known_usage_similarity_flattener, sense_similarity_flattener
        )

        if language == "korean":
            self.embedder = Embedder("klue/bert-base")
        else:
            self.embedder = Embedder("bert-base")

    def get_ranking_with_similarities(
        self,
        target_lemma: str,
        unknown_usage: str,
        known_headwords: List[KnownHeadwordInformation],
    ) -> List[Tuple[int, float]]:

        # Process the unknown usage and the known usages to add [TGT], [/TGT]
        # around the target word before getting embeddings.

        tagged_unknown_usage = self.usage_preprocessor.get_tagged_unknown_usage(
            target_lemma, unknown_usage
        )

        self.usage_preprocessor.tag_headwords_with_targets(
            target_lemma, known_headwords
        )

        # Get embeddings

        definition_similarities = [0 * len(known_headwords)]
        # Should raw definitions be considered?
        if self.definition_weight > 0.0:
            average_token_embedding_for_unknown_usage = (
                self.embedder.get_average_token_embedding(unknown_usage)
            )
            average_token_embeddings_for_headword_sense_definitions = self.embedder.get_average_token_embeddings_for_headword_sense_definitions(
                known_headwords
            )

            definition_similarities = (
                self.similarity_calculator.get_similarities_of_sense_definitions(
                    average_token_embedding_for_unknown_usage,
                    average_token_embeddings_for_headword_sense_definitions,
                )
            )

        usage_similarities = [0 * len(known_headwords)]
        # Should unknown usages be considered?
        if 1 - self.definition_weight > 0.0:

            lemma_embedding_for_unknown_usage = (
                self.embedder.get_embedding_from_tgt_marked_text(tagged_unknown_usage)
            )

            lemma_embeddings_for_headword_sense_known_usages = (
                self.embedder.get_lemma_embeddings_for_headword_sense_known_usages(
                    known_headwords
                )
            )

            usage_similarities = (
                self.similarity_calculator.get_similarities_of_sense_known_usages(
                    lemma_embedding_for_unknown_usage,
                    lemma_embeddings_for_headword_sense_known_usages,
                )
            )

        weighted_similarities = [
            definition_similarities[index] * self.definition_weight
            + usage_similarities[index] * (1 - self.definition_weight)
            for index in range(len(known_headwords))
        ]

        indices_with_sim = [
            (index, weighted_similarities[index])
            for index in range(len(weighted_similarities))
        ]
        indices_with_sim.sort(key=lambda t: t[1], reverse=True)

        return indices_with_sim
