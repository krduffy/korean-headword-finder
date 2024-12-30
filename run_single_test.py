from similarity_comparing import CosineSimilarityStrategy
from test_types import (
    TestCaseForMatchingSenses,
    TestCaseForMatchingKnownUsages,
    UnknownUsageExample,
    Language,
)
import sys
import json

from print_test_result_info import print_test_result_to_stream


def read_from_file_matching_known_usages(
    json_filename: str,
) -> TestCaseForMatchingKnownUsages:

    json_data = None

    try:
        with open(json_filename, "r", encoding="utf-8") as file:
            json_data = json.load(file)
    except IOError:
        raise Exception(
            f"IOError occurred while reading from json file {json_filename}"
        )
    except json.JSONDecodeError:
        raise Exception(f"JSON could not properly decode data in file {json_filename}")

    return TestCaseForMatchingKnownUsages(
        lemma=json_data["lemma"],
        unknown_usage_examples=[
            UnknownUsageExample(
                usage=object["usage"],
                source=object["source"],
                index_of_correct_sense=object["index_of_correct_sense"],
            )
            for object in json_data["unknown_usages"]
        ],
        known_usage_lists=json_data["known_usages"],
    )


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


def do_matching_usage_algorithm(
    language: Language, test_case_data: TestCaseForMatchingKnownUsages
):
    from match_usage_sense_disambiguator import MatchingUsageSenseDisambiguator

    sd = MatchingUsageSenseDisambiguator(language, CosineSimilarityStrategy)

    for unknown_usage_example in test_case_data.unknown_usage_examples:

        res = sd.get_ordered_usage_indices(
            test_case_data.lemma,
            unknown_usage_example.usage,
            test_case_data.known_usage_lists,
        )

        print_test_result_to_stream(
            unknown_usage_example,
            [
                list_of_known_usages[0]
                for list_of_known_usages in test_case_data.known_usage_lists
            ],
            res,
            sys.stdout,
        )


def do_matching_senses_algorithm(args, data):

    from token_weighing import DoNoWeighingStrategy, StopWordsLowWeightStrategy

    strategy = None
    if args[3] == "none":
        strategy = DoNoWeighingStrategy
    elif args[3] == "stop_words_low_weight":
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

    try:
        if filetype == "matching_usage":
            test_case_data = read_from_file_matching_known_usages(filepath)
            do_matching_usage_algorithm(args[1], test_case_data)
        elif filetype == "matching_senses":
            data = read_from_file_matching_senses(filepath)
            do_matching_senses_algorithm(args, data)
    except IOError as error:
        raise IOError(
            f"IO error occurred; file probably doesn't exist. Details: {error}"
        )
