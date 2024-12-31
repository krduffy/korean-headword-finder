from typing import List, Literal
from dataclasses import dataclass

type Language = Literal["english", "korean"]


@dataclass
class UnknownUsageExample:
    usage: str
    source: str
    index_of_correct_sense: int


@dataclass
class KnownSenseInformation:
    definition: str
    known_usages: List[str]


@dataclass(frozen=True)
class KnownHeadwordInformation:
    known_senses: List[KnownSenseInformation]


@dataclass(frozen=True)
class TestCaseForMatchingKnownUsages:
    lemma: str
    unknown_usage_examples: List[UnknownUsageExample]
    known_headwords: List[KnownHeadwordInformation]
