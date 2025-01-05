import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import plotly.express as px


def write_csv(results, columns_in_results, path_to_dir):
    df = pd.DataFrame(results, columns=columns_in_results)
    df.to_csv(f"{path_to_dir}/aggregated.csv", index=False)


def write_pngs(path_to_dir):

    df = pd.read_csv(f"{path_to_dir}/aggregated.csv")

    rc("font", family="New Gulim")
    df = combine_config_columns(df)

    write_all_lemma_files(df, path_to_dir)
    write_aggregated_file(df, path_to_dir)
    write_choice_result_scatter_plot(df.copy(), path_to_dir)


def write_all_lemma_files(df: pd.DataFrame, path_to_dir: str):

    lemmas = df["lemma"].unique()

    for lemma in lemmas:
        os.makedirs(os.path.dirname(f"{path_to_dir}/lemmas/{lemma}.png"), exist_ok=True)

        data_for_lemma_df = df[df["lemma"] == lemma]

        data_for_lemma_df = add_correct_statistics(data_for_lemma_df)

        data_for_lemma_df = data_for_lemma_df.groupby("combined_config")[
            [
                "correct_minus_average_incorrect",
                "correct_minus_best_incorrect",
            ]
        ].mean()

        do_bar_plot(data_for_lemma_df, f"Results for {lemma}")

        plt.savefig(f"{path_to_dir}/lemmas/{lemma}.png")
        plt.close()


def do_bar_plot(df, title: str):
    df.plot(
        kind="bar",
        title=title,
        color=["#2ecc71", "#e74c3c", "#3498db"],
        xlabel="Configuration",
        ylabel="Score",
        figsize=(16, 8),
        rot=30,
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()


def combine_config_columns(df: pd.DataFrame):

    new_df = df.copy()

    def format_config(column_config):
        return (
            f"{column_config['definition_weight']}"
            + f"/{column_config['known_usage_similarity_flattener'][0]}"
            + f"/{column_config['sense_similarity_flattener'][0]}"
            + f"/{column_config['definition_similarity_flattener'][0]}"
        )

    columns_to_combine = [
        "definition_weight",
        "known_usage_similarity_flattener",
        "sense_similarity_flattener",
        "definition_similarity_flattener",
    ]

    new_df["combined_config"] = new_df[columns_to_combine].apply(format_config, axis=1)

    return new_df


def add_correct_statistics(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()

    new_df["correct_minus_average_incorrect"] = df.groupby("combined_config")[
        "correct_minus_average_incorrect"
    ].transform("mean")

    new_df["correct_minus_best_incorrect"] = df.groupby("combined_config")[
        "correct_minus_best_incorrect"
    ].transform("mean")

    new_df["proportion_of_times_correct_top"] = df.groupby("combined_config")[
        "correct_minus_best_incorrect"
    ].transform(lambda item: (item > 0.0).mean())

    return new_df


def write_aggregated_file(df: pd.DataFrame, path_to_dir: str):

    aggregated_df = add_correct_statistics(df)

    os.makedirs(os.path.dirname(f"{path_to_dir}/aggregated.png"), exist_ok=True)

    aggregated_df = aggregated_df.groupby("combined_config")[
        [
            "correct_minus_average_incorrect",
            "correct_minus_best_incorrect",
            "proportion_of_times_correct_top",
        ]
    ].mean()

    do_bar_plot(aggregated_df, "Aggregated Results")

    plt.savefig(f"{path_to_dir}/aggregated.png")
    plt.close()


def combine_config_plus_choice_value_columns(df: pd.DataFrame) -> pd.DataFrame:

    new_df = df.copy()

    def format_column(column_config):
        return (
            f"{column_config['definition_weight']}"
            + f"/{column_config['known_usage_similarity_flattener'][0]}"
            + f"/{column_config['sense_similarity_flattener'][0]}"
            + f"/{column_config['definition_similarity_flattener'][0]}"
            + f"/{column_config['min_acceptance']}"
            + f"/{column_config['min_delta']}"
        )

    columns_to_combine = [
        "definition_weight",
        "known_usage_similarity_flattener",
        "sense_similarity_flattener",
        "definition_similarity_flattener",
        "min_acceptance",
        "min_delta",
    ]

    new_df["combined_config_and_choice_values"] = new_df[columns_to_combine].apply(
        format_column, axis=1
    )

    return new_df


def add_choice_result_statistics(df: pd.DataFrame) -> pd.DataFrame:

    new_df = df.copy()

    new_df["proportion_of_times_choice_made"] = df.groupby(
        "combined_config_and_choice_values"
    )["choice_result"].transform(lambda choice_result: (choice_result != 0).mean())

    new_df["proportion_of_choices_correct"] = df.groupby(
        "combined_config_and_choice_values"
    )["choice_result"].transform(
        lambda choice_result: (choice_result == 1).sum()
        / ((choice_result == 1).sum() + (choice_result == -1).sum())
    )

    return new_df


def write_choice_result_scatter_plot(df: pd.DataFrame, path_to_dir: str):

    df = combine_config_plus_choice_value_columns(df)
    df = add_choice_result_statistics(df)

    # has no labels
    df.plot(
        kind="scatter",
        x="proportion_of_times_choice_made",
        y="proportion_of_choices_correct",
    )
    plt.savefig(f"{path_to_dir}/choice_result_scatter_plot.png")
    plt.close()

    plot = px.scatter(
        df,
        x="proportion_of_times_choice_made",
        y="proportion_of_choices_correct",
        hover_data="combined_config_and_choice_values",
        title="Choice Results",
    )

    plot.write_html(f"{path_to_dir}/choice_result_scatter_plot.html")


if __name__ == "__main__":
    write_pngs("test_results/kor/second-iteration")
