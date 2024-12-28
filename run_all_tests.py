import sys
from typing import List, Tuple
from test_types import Language, Example
import os
from run_single_test import read_from_file
from itertools import product
from sense_disambiguator import SenseDisambiguator
from token_weighting import DoNoWeightingStrategy, StopWordsLowWeightStrategy
from write_test_data_to_csv import write_test_data_to_csv

all_configs = [[DoNoWeightingStrategy, StopWordsLowWeightStrategy]]
config_combinations = product(*all_configs)


def run_all_examples_with_all_configs(
    example_sense_pairs: List[Tuple[List[Example], List[str]]], language: Language
):

    results = []

    for config_combination in config_combinations:
        sense_disambiguator = SenseDisambiguator(language, *config_combination)

        for example_sense_pair in example_sense_pairs:
            for example in example_sense_pair[0]:
                res = sense_disambiguator.get_ordered_sense_indices(
                    example[0], example_sense_pair[1]
                )

                results.append(
                    {
                        "token weighting strategy": config_combination[0],
                        "senses": example_sense_pairs[1],
                        "examples": {
                            "input example": example[0],
                            "source": example[1],
                            "correct sense": example[2],
                            "sense_similarities": res,
                        },
                    }
                )

    return results


def run_all_in_dir(dir: str, language: Language):
    all_files = os.listdir(dir)

    all_files = [os.path.join(dir, filename) for filename in all_files]
    example_sense_pairs = [read_from_file(filename) for filename in all_files]

    return run_all_examples_with_all_configs(example_sense_pairs, language)


def run_korean_tests():
    results = run_all_in_dir("inputs/kor/homographs", "korean")
    write_test_data_to_csv(results, "test_results/kor/homographs.csv")


def run_english_tests():
    results = run_all_in_dir("inputs/eng/homographs", "english")
    write_test_data_to_csv(results, "test_results/eng/homographs.csv")


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
