from typing import List, TextIO, Tuple
from test_types import Example

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
    example: Example,
    senses: List[str],
    index_ranking: List[Tuple[int, float]],
    file: TextIO,
):

    print(f"Example: {example[0]} | source: {example[1]}", file=file)

    for sensenum, similarity in index_ranking:
        formatted_sense = (
            f"{GREEN}{senses[sensenum]}{RESET}"
            if sensenum == example[2]
            else senses[sensenum]
        )

        formatted_similarity = (
            f"{get_similarity_score_color(similarity)}{similarity:.5f}{RESET}"
        )

        print(f"{formatted_similarity} | {formatted_sense}", file=file)
