from test_types import TestCase
import os
import sys

from print_test_result_info import print_test_result_to_stream


def read_from_file(filename: str) -> TestCase:

    all_lines = []

    with open(filename, "r", encoding="utf-8") as file:
        all_lines = [line.strip() for line in file]

    current_index = 0

    num_senses = int(all_lines[current_index])
    current_index += 1

    senses = all_lines[current_index : current_index + num_senses]
    current_index += num_senses

    num_examples = int(all_lines[current_index])
    current_index += 1
    examples = []

    for _ in range(num_examples):
        examples.append(
            (
                all_lines[current_index],
                all_lines[current_index + 1],
                int(all_lines[current_index + 2]),
            )
        )
        current_index += 3

    return (examples, senses)


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 4:
        raise ValueError(
            """Usage: `python run_single_test.py <language> <word> <weighing policy class name>`"""
        )

    filename = ""
    word = args[2]

    english_dir = "eng_inputs/homographs"
    korean_dir = "kor_inputs/homographs"

    if args[1] == "english":
        filename = os.path.join(english_dir, f"{word}.txt")
    elif args[1] == "korean":
        filename = os.path.join(korean_dir, f"{word}.txt")
    else:
        raise ValueError("language passed must be 'english' or 'korean'.")

    try:
        examples, senses = read_from_file(filename)
    except IOError as error:
        raise IOError(
            f"IO error occurred; file probably doesn't exist. Details: {error}"
        )

    from token_weighting import DoNoWeightingStrategy, StopWordsLowWeightStrategy

    strategy = None
    if args[3] == "none":
        strategy = DoNoWeightingStrategy
    elif args[3] == "stop_words_low_weight":
        strategy = StopWordsLowWeightStrategy
    else:
        raise ValueError(
            """weighing policy class name passed must be
                         'none' or 'stop_words_low_weight'. """
        )

    from sense_disambiguator import SenseDisambiguator

    sd = SenseDisambiguator(args[1], strategy)

    for example in examples:
        index_ranking = sd.get_ordered_sense_indices(example[0], senses)
        print_test_result_to_stream(example, senses, index_ranking, sys.stdout)
