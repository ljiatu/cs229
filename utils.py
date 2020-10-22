from typing import Tuple

import pandas as pd


def load_data(feature_file_path: str, score_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads pre-computed features and cleavage efficiency scores from the specified paths.

    The first column of both the feature_df and score_df contains labels for the target sequences.
    """
    feature_df = pd.read_csv(feature_file_path)
    score_df = pd.read_csv(score_file_path)

    return feature_df, score_df
