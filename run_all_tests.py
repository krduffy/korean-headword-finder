from match_usage_sense_disambiguator import MatchingUsageHeadwordDisambiguator
from similarity_flattener import MaxStrategy, AverageStrategy
from test_types import Language, TestCaseForMatchingKnownUsages
import sys
from typing import List, Tuple
import os
from run_single_test import read_from_file
from itertools import product
from write_result_files import write_result_files

all_configs = [
    # Definition weights tested
    [0.0],
    # Known usage similarity flatteners tested
    [MaxStrategy],
    # Sense similarity flatteners tested
    [MaxStrategy],
    # Definition similarity flatteners tested
    [MaxStrategy],
]
config_combinations = product(*all_configs)


def get_score(
    correct_headword_id: int, headword_similarities: List[Tuple[int, float]]
) -> float:

    correct_headword_similarity = headword_similarities[correct_headword_id][1]

    sum_of_incorrect = (
        sum([headword_similarity[1] for headword_similarity in headword_similarities])
    ) - correct_headword_similarity

    average_of_incorrect = sum_of_incorrect / (len(headword_similarities) - 1)

    return correct_headword_similarity - average_of_incorrect


def run_all_examples_with_all_configs(
    test_cases: List[TestCaseForMatchingKnownUsages],
    language: Language,
):
    results = []

    columns_in_results = [
        "lemma",
        "definition_weight",
        "known_usage_similarity_flattener",
        "sense_similarity_flattener",
        "definition_similarity_flattener",
        "score",
    ]

    for config_combination in config_combinations:
        sense_disambiguator = MatchingUsageHeadwordDisambiguator(
            language, *config_combination
        )

        for test_case in test_cases:

            for unknown_usage_example in test_case.unknown_usage_examples:

                this_examples_similarities = (
                    sense_disambiguator.get_ranking_with_similarities(
                        test_case.lemma,
                        unknown_usage_example.usage,
                        test_case.known_headwords,
                    )
                )

                results.append(
                    [
                        test_case.lemma,
                        config_combination[0],
                        config_combination[1].__name__,
                        config_combination[2].__name__,
                        config_combination[3].__name__,
                        get_score(
                            unknown_usage_example.index_of_correct_headword,
                            this_examples_similarities,
                        ),
                    ]
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
    write_result_files(results, columns_in_results, "test_results/kor")


def run_english_tests():
    results, columns_in_results = run_all_in_dir("inputs/eng", "english")
    write_result_files(results, columns_in_results, "test_results/eng")


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
