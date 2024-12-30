from similarity_comparing import CosineSimilarityStrategy
from test_types import TestCaseForMatchingSenses, TestCaseForMatchingUsage
import os
import sys

from print_test_result_info import print_test_result_to_stream


def read_from_file_matching_usage(filename: str) -> TestCaseForMatchingUsage:

    all_lines = []

    with open(filename, "r", encoding="utf-8") as file:
        all_lines = [line.strip() for line in file]

    current_index = 0

    num_senses = int(all_lines[current_index])
    current_index += 1

    list_of_sense_usages = []

    for _ in range(num_senses):
        num_of_usages = int(all_lines[current_index])
        current_index += 1

        these_usages = all_lines[current_index : current_index + num_of_usages]
        current_index += num_of_usages
        list_of_sense_usages.append(these_usages)

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

    return (list_of_sense_usages, examples)


def read_from_file_matching_senses(filename: str) -> TestCaseForMatchingSenses:

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


def do_matching_usage_algorithm(args, data):
    from match_usage_sense_disambiguator import MatchingUsageSenseDisambiguator

    sd = MatchingUsageSenseDisambiguator(args[1], CosineSimilarityStrategy)

    print(sd.get_ordered_usage_indices("눈", data[1][2][0], data[0]))


def do_matching_senses_algorithm(args, data):

    from token_weighing import DoNoWeighingStrategy, StopWordsLowWeightStrategy

    strategy = None
    if args[4] == "none":
        strategy = DoNoWeighingStrategy
    elif args[4] == "stop_words_low_weight":
        strategy = StopWordsLowWeightStrategy
    else:
        raise ValueError(
            """weighing policy class name passed must be
                         'none' or 'stop_words_low_weight'."""
        )

    from sense_disambiguator import SenseDisambiguator

    sd = SenseDisambiguator(args[1], strategy, CosineSimilarityStrategy)

    for example in data[1]:
        index_ranking = sd.get_ordered_sense_indices(example[0], data[0])
        print_test_result_to_stream(example, data[0], index_ranking, sys.stdout)


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 4:
        raise ValueError(
            """Usage: `python run_single_test.py <language> <filepath> <weighing policy class name> <match usage>`"""
        )

    filepath = args[2]

    filetype = args[4]

    if filetype == "matching_usage":
        data = read_from_file_matching_usage(filepath)
        do_matching_usage_algorithm(args, data)
    elif filetype == "senses":
        data = read_from_file_matching_senses(filepath)
        do_matching_senses_algorithm(args, data)

    try:
        if filetype == "matching_usage":
            data = read_from_file_matching_usage(filepath)
            do_matching_usage_algorithm(args, data)
        elif filetype == "senses":
            data = read_from_file_matching_senses(filepath)
            do_matching_senses_algorithm(args, data)
    except IOError as error:
        raise IOError(
            f"IO error occurred; file probably doesn't exist. Details: {error}"
        )
