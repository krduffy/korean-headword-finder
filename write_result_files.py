import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import os


def write_csv(results, columns_in_results, path_to_dir):
    df = pd.DataFrame(results, columns=columns_in_results)
    df.to_csv(f"{path_to_dir}/aggregated.csv", index=False)


def write_pngs(path_to_dir):

    df = pd.read_csv(f"{path_to_dir}/aggregated.csv")

    rc("font", family="New Gulim")
    df = combine_config_columns(df)

    write_all_lemma_files(df, path_to_dir)
    write_aggregated_file(df, path_to_dir)


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
                "proportion_of_times_correct_top",
            ]
        ].mean()

        data_for_lemma_df.plot(
            kind="bar",
            title=f"Results for '{lemma}'",
            color=["blue", "orange", "purple"],
            xlabel="Configuration",
            ylabel="Score",
            figsize=(16, 8),
        )
        plt.savefig(f"{path_to_dir}/lemmas/{lemma}.png")
        plt.close()


def combine_config_columns(df: pd.DataFrame):

    new_df = df

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
    new_df = df

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

    aggregated_df.plot(
        kind="bar",
        title="Aggregated Results",
        color=["blue", "orange", "purple"],
        xlabel="Configuration",
        ylabel="Values",
        figsize=(16, 8),
    )

    plt.savefig(f"{path_to_dir}/aggregated.png")
    plt.close()


if __name__ == "__main__":
    write_pngs("test_results/kor")
