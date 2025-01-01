from typing import List
from embedder import Embedder
from test_types import KnownHeadwordInformation, Language
from usage_preprocessor import UsagePreprocessor


class MatchingUsageHeadwordDisambiguator:

    def __init__(
        self,
        language: Language,
    ):

        if language == "korean":
            self.embedder = Embedder("klue/bert-base")
            from lemmatizer import KoreanLemmatizer

            self.usage_preprocessor = UsagePreprocessor(
                KoreanLemmatizer(attach_다_to_verbs=True)
            )
        else:
            self.embedder = Embedder("bert-base")
            from lemmatizer import EnglishLemmatizer

            self.usage_preprocessor = UsagePreprocessor(EnglishLemmatizer())

    def get_all_embeddings(
        self,
        target_lemma: str,
        unknown_usage: str,
        known_headwords: List[KnownHeadwordInformation],
    ):

        # Process the unknown usage and the known usages to add [TGT], [/TGT]
        # around the target word before getting embeddings.

        tagged_unknown_usage = self.usage_preprocessor.get_tagged_unknown_usage(
            target_lemma, unknown_usage
        )

        self.usage_preprocessor.tag_headwords_with_targets(
            target_lemma, known_headwords
        )

        # Get embeddings

        average_token_embedding_for_unknown_usage = (
            self.embedder.get_average_token_embedding(unknown_usage)
        )
        average_token_embeddings_for_headword_sense_definitions = (
            self.embedder.get_average_token_embeddings_for_headword_sense_definitions(
                known_headwords
            )
        )

        lemma_embedding_for_unknown_usage = (
            self.embedder.get_embedding_from_tgt_marked_text(tagged_unknown_usage)
        )

        lemma_embeddings_for_headword_sense_known_usages = (
            self.embedder.get_lemma_embeddings_for_headword_sense_known_usages(
                known_headwords
            )
        )

        return (
            average_token_embedding_for_unknown_usage,
            average_token_embeddings_for_headword_sense_definitions,
            lemma_embedding_for_unknown_usage,
            lemma_embeddings_for_headword_sense_known_usages,
        )
