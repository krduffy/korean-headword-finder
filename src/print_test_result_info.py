from typing import List, TextIO, Tuple
from headword_chooser import choose_headword
from test_types import UnknownUsageExample
from get_score import get_correct_minus_avg_incorrect

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def get_similarity_score_color(similarity_score) -> str:
    if similarity_score < 0.4:
        return RED
    if similarity_score < 0.7:
        return YELLOW

    return GREEN


def print_test_result_to_stream(
    example: UnknownUsageExample,
    headwords: List[str],
    index_ranking: List[Tuple[int, float]],
    file: TextIO,
):

    print("=" * 70, file=file)
    print(f"Unknown usage: {example.usage} | source: {example.source}", file=file)

    for headword_num, similarity in index_ranking:
        formatted_sense = (
            f"{GREEN}{headwords[headword_num]}{RESET}"
            if headword_num == example.index_of_correct_headword
            else headwords[headword_num]
        )

        formatted_similarity = (
            f"{get_similarity_score_color(similarity)}{similarity:.5f}{RESET}"
        )

        print(f"{formatted_similarity} | {formatted_sense}", file=file)

    print(
        f"\nCorrect - avg incorrect is {get_correct_minus_avg_incorrect(example.index_of_correct_headword, index_ranking)}"
    )

    chosen = choose_headword(index_ranking, min_acceptance=0.5, min_delta=0.05)

    if chosen is not None:
        was_correct = chosen == example.index_of_correct_headword
        print(
            f"{GREEN if was_correct else RED}Chosen index (min acceptance 0.5, min delta 0.05) is {chosen}{RESET}"
        )
    else:
        print(
            f"{YELLOW}Could not make any choice (min acceptance 0.5, min delta 0.05){RESET}"
        )

    print("=" * 70, file=file)
