from typing import List
from test_types import (
    TestCaseForMatchingKnownUsages,
    UnknownUsageExample,
    Language,
    KnownHeadwordInformation,
)
import sys
from similarity_flattener import AverageStrategy, MaxStrategy
import json
import re

from print_test_result_info import print_test_result_to_stream


def delete_sources_from_usages(
    known_headwords: List[KnownHeadwordInformation],
) -> None:

    for headword in known_headwords:
        for sense in headword["known_senses"]:

            sense["known_usages"] = [
                re.sub(r"≪.*≫", "", usage) for usage in sense["known_usages"]
            ]


def read_from_file(
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

    delete_sources_from_usages(json_data["known_headwords"])

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
    disambiguator_args, test_case_data: TestCaseForMatchingKnownUsages
):
    from match_usage_sense_disambiguator import MatchingUsageHeadwordDisambiguator

    sd = MatchingUsageHeadwordDisambiguator(*disambiguator_args)

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


def string_to_flattening_strategy(string: str):
    if string == "max":
        return MaxStrategy
    if string == "average":
        return AverageStrategy
    raise ValueError(f"Unknown strategy supplied '{string}'")


if __name__ == "__main__":

    args = sys.argv

    if len(args) < 7:
        raise ValueError(
            """Usage: `python run_single_test.py 
            <language> 
            <filepath> 
            <definition_weight>
            <known_usage_similarity_flattener>
            <sense_similarity_flattener>
            <definition_similarity_flatten>`"""
        )

    if args[1] not in ["korean", "english"]:
        raise ValueError(f"Unknown language supplied '{args[1]}'")

    disambiguator_args = [
        args[1],
        float(args[3]),
        *[string_to_flattening_strategy(string) for string in args[4:7]],
    ]

    print(disambiguator_args)

    filepath = args[2]

    test_case_data = read_from_file(filepath)
    do_matching_usage_algorithm(disambiguator_args, test_case_data)
