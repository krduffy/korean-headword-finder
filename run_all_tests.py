import sys
from typing import List, Tuple
from similarity_comparing import CosineSimilarityStrategy, EuclideanDistanceStrategy
from test_types import Language, Example
import os
from run_single_test import read_from_file
from itertools import product
from sense_disambiguator import SenseDisambiguator
from token_weighing import DoNoWeighingStrategy, StopWordsLowWeightStrategy
from write_result_files import write_result_files

all_configs = [
    [DoNoWeighingStrategy, StopWordsLowWeightStrategy],
    [
        CosineSimilarityStrategy,
        # EuclideanDistanceStrategy
    ],
]
config_combinations = product(*all_configs)


def run_all_examples_with_all_configs(
    example_sense_pairs: List[Tuple[List[Example], List[str]]],
    language: Language,
    target_lemmas: List[str],
):

    results = []

    for config_combination in config_combinations:
        sense_disambiguator = SenseDisambiguator(language, *config_combination)

        for example_sense_pair, target_lemma in zip(example_sense_pairs, target_lemmas):

            this_examples_results = [
                sense_disambiguator.get_ordered_sense_indices(
                    example[0], example_sense_pair[1]
                )
                for example in example_sense_pair[0]
            ]

            results.append(
                {
                    "target_lemma": target_lemma,
                    "token_weighing_strategy": config_combination[0].__name__,
                    "similarity_comparison_strategy": config_combination[1].__name__,
                    "senses": example_sense_pairs[1],
                    "examples_and_sense_similarities": [
                        {
                            "example": example,
                            "sense_similarity_results": sense_similarity_results,
                        }
                        for (example, sense_similarity_results) in zip(
                            example_sense_pair[0], this_examples_results
                        )
                    ],
                }
            )

    return results


def get_all_files_starting_in_dir(dir: str):
    all_files = []

    for file_or_dir in os.listdir(dir):

        joined_with_path = os.path.join(dir, file_or_dir)

        if os.path.isdir(joined_with_path):
            all_files.extend(get_all_files_starting_in_dir(joined_with_path))
        else:
            all_files.append(joined_with_path)

    return all_files


def run_all_in_dir(dir: str, language: Language):

    all_files = get_all_files_starting_in_dir(dir)
    target_lemmas = [
        os.path.splitext(filename)[0].split(os.sep)[-1] for filename in all_files
    ]

    example_sense_pairs = [read_from_file(filename) for filename in all_files]

    return run_all_examples_with_all_configs(
        example_sense_pairs, language, target_lemmas
    )


def run_korean_tests():
    results = run_all_in_dir("inputs/kor/homographs", "korean")
    write_result_files(results, "test_results/kor/homographs")


def run_english_tests():
    results = run_all_in_dir("inputs/eng/homographs", "english")
    write_result_files(results, "test_results/eng/homographs")


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
