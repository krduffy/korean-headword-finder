from headword_chooser import choose_headword
from headword_ranker import rank_headwords
from match_usage_sense_disambiguator import MatchingUsageHeadwordDisambiguator
from similarity_calculator import SimilarityCalculator
from similarity_flattener import MaxStrategy, AverageStrategy
from test_types import Language, TestCaseForMatchingKnownUsages
import sys
from typing import List
import os
from run_single_test import read_from_file
from itertools import product
from write_result_files import write_csv
from get_score import get_correct_minus_avg_incorrect, get_correct_minus_best_incorrect

all_configs = [
    # Definition weights tested
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # Known usage similarity flatteners tested
    [MaxStrategy],
    # Sense similarity flatteners tested
    [MaxStrategy, AverageStrategy],
    # Definition similarity flatteners tested
    [MaxStrategy, AverageStrategy],
]
config_combinations = list(product(*all_configs))

all_choice_values = [
    [0.3, 0.4, 0.5, 0.6],
    [0.0, 0.025, 0.05, 0.075, 0.1],
]
choice_values_combinations = list(product(*all_choice_values))

GREEN = "\033[32m"
RESET = "\033[0m"


def run_all_examples_with_all_configs(
    test_cases: List[TestCaseForMatchingKnownUsages],
    language: Language,
):
    results = []

    sense_disambiguator = MatchingUsageHeadwordDisambiguator(language)

    columns_in_results = [
        "lemma",
        "definition_weight",
        "known_usage_similarity_flattener",
        "known_usage_second_similarity_flattener",
        "definition_similarity_flattener",
        "correct_minus_average_incorrect",
        "correct_minus_best_incorrect",
        "min_acceptance",
        "min_delta",
        "choice_result",
    ]

    for test_case in test_cases:

        for unknown_usage_example in test_case.unknown_usage_examples:

            this_example_embeddings = sense_disambiguator.get_all_embeddings(
                test_case.lemma, unknown_usage_example.usage, test_case.known_headwords
            )

            for config_combination in config_combinations:

                this_examples_similarities = rank_headwords(
                    len(test_case.known_headwords),
                    this_example_embeddings,
                    config_combination[0],
                    SimilarityCalculator(*config_combination[1:]),
                )

                correct_minus_avg_incorrect = get_correct_minus_avg_incorrect(
                    unknown_usage_example.index_of_correct_headword,
                    this_examples_similarities,
                )
                correct_minus_best_incorrect = get_correct_minus_best_incorrect(
                    unknown_usage_example.index_of_correct_headword,
                    this_examples_similarities,
                )

                for choice_values in choice_values_combinations:
                    chosen_index = choose_headword(
                        this_examples_similarities,
                        min_acceptance=choice_values[0],
                        min_delta=choice_values[1],
                    )

                    if chosen_index is None:
                        choice_result = 0
                    elif (
                        chosen_index == unknown_usage_example.index_of_correct_headword
                    ):
                        choice_result = 1
                    else:
                        choice_result = -1

                    results.append(
                        [
                            test_case.lemma,
                            config_combination[0],
                            config_combination[1].__name__,
                            config_combination[2].__name__,
                            config_combination[3].__name__,
                            correct_minus_avg_incorrect,
                            correct_minus_best_incorrect,
                            choice_values[0],
                            choice_values[1],
                            choice_result,
                        ]
                    )

        print(
            f"{GREEN}Finished running simulations for lemma {test_case.lemma}.{RESET}"
        )

    return results, columns_in_results


def get_all_files_starting_in_dir(dir: str):
    all_files = []

    for file_or_dir in os.listdir(dir):

        joined_with_path = os.path.join(dir, file_or_dir)

        if os.path.isdir(joined_with_path):
            all_files.extend(get_all_files_starting_in_dir(joined_with_path))
        elif os.path.splitext(file_or_dir)[1] == ".json":
            all_files.append(joined_with_path)

    return all_files


def run_all_in_dir(dir: str, language: Language):

    all_files = get_all_files_starting_in_dir(dir)

    list_of_test_cases = [read_from_file(filename) for filename in all_files]

    return run_all_examples_with_all_configs(list_of_test_cases, language)


def run_korean_tests():
    results, columns_in_results = run_all_in_dir("inputs/kor", "korean")
    write_csv(results, columns_in_results, "test_results/kor")


def run_english_tests():
    results, columns_in_results = run_all_in_dir("inputs/eng", "english")
    write_csv(results, columns_in_results, "test_results/eng")


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 2:
        raise ValueError("must pass in a language as the first argument.")

    lang = args[1]
    if lang not in ["english", "korean", "all"]:
        raise ValueError(
            "language passed must be one of 'english', 'korean', or 'all'."
        )

    if lang == "english":
        run_english_tests()

    elif lang == "korean":
        run_korean_tests()
