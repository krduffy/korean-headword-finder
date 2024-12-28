import pandas as pd
import os


def write_test_data_to_csv(test_results, csv_path):

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df = pd.DataFrame(test_results)

    df.to_csv(csv_path)
