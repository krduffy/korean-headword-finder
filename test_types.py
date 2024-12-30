from typing import List, Literal
from dataclasses import dataclass

type Language = Literal["english", "korean"]


@dataclass(frozen=True)
class UnknownUsageExample:
    usage: str
    source: str
    index_of_correct_sense: int


@dataclass(frozen=True)
class TestCaseForMatchingSenses:
    lemma: str
    unknown_usage_examples: List[UnknownUsageExample]
    senses: List[str]


@dataclass(frozen=True)
class TestCaseForMatchingKnownUsages:
    lemma: str
    unknown_usage_examples: List[UnknownUsageExample]
    known_usage_lists: List[List[str]]
