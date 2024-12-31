import re
from typing import List
from lemmatizer import Lemmatizer
from test_types import KnownHeadwordInformation


class UsagePreprocessor:

    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer

    def get_tagged_unknown_usage(
        self,
        target_lemma: str,
        unknown_usage: str,
    ):
        return self._find_lemma_and_mark_with_tgt(unknown_usage, target_lemma)

    def tag_headwords_with_targets(
        self,
        target_lemma: str,
        known_headwords: List[KnownHeadwordInformation],
    ) -> None:
        """Mutates known_headwords in place."""

        self._replace_curly_in_all_known_usages(known_headwords, target_lemma)

    def _find_lemma_and_mark_with_tgt(self, text: str, target_lemma: str) -> str:
        """Finds the lemma that is to be disambiguated. If it does not exist
        in the string, then `None` is returned."""

        index_of_target_lemma = self.lemmatizer.find_index_of_lemma(text, target_lemma)

        if index_of_target_lemma == -1:
            # if cannot find lemma exactly, the whole text is target
            return "[TGT]" + text + "[/TGT]"

        individual_tokens = text.split(" ")

        with_replaced = [
            "[TGT]" + token + "[/TGT]" if i == index_of_target_lemma else token
            for i, token in enumerate(individual_tokens)
        ]

        return " ".join(with_replaced)

    def _replace_curly_with_tgt(self, text: str, target_lemma: str) -> str:
        pattern = r"\{(.*)?\}"
        replacer = re.compile(pattern)

        replaced = replacer.sub(r"[TGT]\1[/TGT]", text)

        # saw a change
        if text != replaced:
            return replaced

        return self._find_lemma_and_mark_with_tgt(text, target_lemma)

    def _replace_curly_in_all_known_usages(
        self, known_headwords: List[KnownHeadwordInformation], target_lemma: str
    ) -> None:

        for headword in known_headwords:
            for sense in headword["known_senses"]:

                sense["known_usages"] = [
                    self._replace_curly_with_tgt(usage, target_lemma)
                    for usage in sense["known_usages"]
                ]
