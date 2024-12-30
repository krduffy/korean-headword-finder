import re
from typing import List, Tuple
from torch import Tensor, no_grad
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from lemmatizer import EnglishLemmatizer, KoreanLemmatizer
from similarity_comparing import SimilarityComparisonStrategy
from test_types import Language


class MatchingUsageSenseDisambiguator:

    def __init__(
        self,
        language: Language,
        similarity_comparison_strategy: SimilarityComparisonStrategy,
    ):
        pretrained_model = ""

        if language == "korean":
            pretrained_model = "klue/bert-base"
            self.lemmatizer = KoreanLemmatizer(attach_다_to_verbs=True)

        elif language == "english":
            pretrained_model = "bert-base-uncased"
            self.lemmatizer = EnglishLemmatizer()

        else:
            raise ValueError("`language` must be 'english' or 'korean'.")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        added_target_word_tokens = {"additional_special_tokens": ["[TGT]", "[/TGT]"]}
        self.tokenizer.add_special_tokens(added_target_word_tokens)

        self.model = BertModel.from_pretrained(pretrained_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.similarity_comparer = similarity_comparison_strategy()

    def _get_embedding_from_tgt_marked_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        tokens = self.tokenizer.tokenize(text)

        with no_grad():
            outputs = self.model(**inputs)

        start_idx = tokens.index("[TGT]") + 1
        end_idx = tokens.index("[/TGT]")

        hidden_states = outputs.last_hidden_state[0]
        tagged_embeddings = hidden_states[start_idx:end_idx]

        return torch.mean(tagged_embeddings, dim=0)

    def _find_lemma_and_mark_with_tgt(self, text: str, target_lemma: str) -> str:
        """Finds the lemma that is to be disambiguated. If it does not exist
        in the string, then `None` is returned."""

        index_of_target_lemma = self.lemmatizer.find_index_of_lemma(text, target_lemma)

        if index_of_target_lemma == -1:
            return None

        individual_tokens = text.split(" ")

        with_replaced = [
            "[TGT]" + token + "[/TGT]" if i == index_of_target_lemma else token
            for i, token in enumerate(individual_tokens)
        ]

        return " ".join(with_replaced)

    def _replace_curly_with_tgt(self, text: str) -> str:
        pattern = r"\{(.*)?\}"
        replacer = re.compile(pattern)

        return replacer.sub(r"[TGT]\1[/TGT]", text)

    def _replace_curly_in_all_known_usages(
        self, known_usages: List[List[str]]
    ) -> List[List[str]]:

        return [
            [
                self._replace_curly_with_tgt(still_with_curly)
                for still_with_curly in set_of_usages
            ]
            for set_of_usages in known_usages
        ]

    def _get_all_embeddings_for_known_usages(
        self, known_usages: List[List[str]]
    ) -> List[List[float]]:
        return [
            [
                self._get_embedding_from_tgt_marked_text(known_usage_text)
                for known_usage_text in set_of_usages
            ]
            for set_of_usages in known_usages
        ]

    def _get_all_similarities(
        self,
        unknown_usage_embedding: Tensor,
        known_usage_embeddings: List[List[Tensor]],
    ) -> List[List[float]]:
        unknown_norm = F.normalize(unknown_usage_embedding, p=2, dim=-1)

        return [
            [
                self.similarity_comparer.get_similarity(
                    unknown_usage_embedding, known_usage_embedding.unsqueeze(0)
                )
                for known_usage_embedding in set_of_embeddings
            ]
            for set_of_embeddings in known_usage_embeddings
        ]

    def get_ordered_usage_indices(
        self, target_lemma: str, unknown_usage: str, known_usages: List[List[str]]
    ) -> List[Tuple[int, float]]:

        tgt_marked_unknown_usage = self._find_lemma_and_mark_with_tgt(
            unknown_usage, target_lemma
        )

        if tgt_marked_unknown_usage is None:
            print("Could not find lemma in input `unknown_usage`.")
            return []

        tgt_marked_known_usages = self._replace_curly_in_all_known_usages(known_usages)

        embedding_of_target_lemma_in_unknown = self._get_embedding_from_tgt_marked_text(
            tgt_marked_unknown_usage
        )
        embeddings_of_target_lemma_in_knowns = (
            self._get_all_embeddings_for_known_usages(tgt_marked_known_usages)
        )

        similarities = self._get_all_similarities(
            embedding_of_target_lemma_in_unknown, embeddings_of_target_lemma_in_knowns
        )

        # flattening to one level; will later add a configurable way to do this
        similarities = [
            sum(list_of_similarities) / len(list_of_similarities)
            for list_of_similarities in similarities
        ]

        indices_with_sim = [
            (index, similarities[index]) for index in range(len(similarities))
        ]
        indices_with_sim.sort(key=lambda t: t[1], reverse=True)

        return indices_with_sim
