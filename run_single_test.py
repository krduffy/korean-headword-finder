from test_types import (
    TestCaseForMatchingKnownUsages,
    UnknownUsageExample,
    Language,
)
import sys
from similarity_flattener import AverageStrategy
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
                index_of_correct_headword=object["index_of_correct_headword"],
            )
            for object in json_data["unknown_usage_examples"]
        ],
        known_headwords=json_data["known_headwords"],
    )


def do_matching_usage_algorithm(
    language: Language, test_case_data: TestCaseForMatchingKnownUsages
):
    from match_usage_sense_disambiguator import MatchingUsageHeadwordDisambiguator

    sd = MatchingUsageHeadwordDisambiguator(
        language, 0.5, AverageStrategy, AverageStrategy, AverageStrategy
    )

    for unknown_usage_example in test_case_data.unknown_usage_examples:

        res = sd.get_ranking_with_similarities(
            test_case_data.lemma,
            unknown_usage_example.usage,
            test_case_data.known_headwords,
        )

        print_test_result_to_stream(
            unknown_usage_example,
            [
                headword["known_senses"][0]["definition"]
                for headword in test_case_data.known_headwords
            ],
            res,
            sys.stdout,
        )


if __name__ == "__main__":

    args = sys.argv

    # if len(args) < 4:
    #    raise ValueError(
    #        """Usage: `python run_single_test.py <language> <filepath> <weighing policy class name> <match usage>`"""
    #    )

    filepath = "inputs/kor/matching_usage/ignore/임종.json"

    test_case_data = read_from_file_matching_known_usages(filepath)
    do_matching_usage_algorithm(args[1], test_case_data)
