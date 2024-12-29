import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os


def get_score(example_and_sense_similarity_set):

    ex = example_and_sense_similarity_set["example"]
    ssr = example_and_sense_similarity_set["sense_similarity_results"]

    corrects = [ssr[i][1] for i in range(len(ssr)) if ssr[i][0] == ex[2]]
    incorrects = [ssr[i][1] for i in range(len(ssr)) if ssr[i][0] != ex[2]]

    return np.mean(corrects) - np.mean(incorrects)


def get_rows_for_test_result(test_result) -> list:
    return [
        [
            test_result["target_lemma"],
            test_result["token_weighing_strategy"]
            + " / "
            + test_result["similarity_comparison_strategy"],
            example_and_sense_set["example"][0],
            example_and_sense_set["example"][1],
            get_score(example_and_sense_set),
        ]
        for example_and_sense_set in test_result["examples_and_sense_similarities"]
    ]


def get_dataframe(test_results) -> pd.DataFrame:

    flattened = [
        row
        for test_result in test_results
        for row in get_rows_for_test_result(test_result)
    ]

    return pd.DataFrame(
        flattened,
        columns=[
            "target_lemma",
            "strategies",
            "example_text",
            "example_source",
            "score",
        ],
    )


def write_result_files(test_results, path_to_file_without_ext):

    os.makedirs(os.path.dirname(f"{path_to_file_without_ext}.png"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{path_to_file_without_ext}.csv"), exist_ok=True)

    df = get_dataframe(test_results)

    create_png(df, f"{path_to_file_without_ext}.png")


def create_png(df: pd.DataFrame, png_path: str):

    rc("font", family="New Gulim")

    lemmas = df["target_lemma"].unique()
    strategies_values = df["strategies"].unique()

    plt.figure(figsize=(12, 6))

    x = np.arange(len(lemmas))
    width = 0.7 / len(strategies_values)

    for i, strategy_set in enumerate(strategies_values):
        strategy_data = df[df["strategies"] == strategy_set]
        scores = [
            strategy_data[strategy_data["target_lemma"] == lemma]["score"].iloc[0]
            for lemma in lemmas
        ]

        positions = x + (i - 0.5) * width
        plt.bar(positions, scores, width, label=strategy_set)

    plt.xlabel("Lemmas")
    plt.ylabel("Score")
    plt.title("Comparison of Strategies Across Lemmas")
    plt.xticks(x, lemmas)
    plt.legend()

    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
