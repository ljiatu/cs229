from sklearn.model_selection import train_test_split

from utils import load_data

TEST_SET_RATIO = 0.2


def split(feature_file_path: str, score_file_path: str) -> None:
    """
    Splits the dataset into training and test data, and save them into two files.

    This is to ensure different models and different experiments run on the same training and test data.
    """
    feature_df, score_df = load_data(feature_file_path, score_file_path)
    train_feature_df, test_feature_df, train_score_df, test_score_df = train_test_split(
        feature_df, score_df, test_size=TEST_SET_RATIO, random_state=0,
    )

    feature_file_name = feature_file_path.split(".")[0]
    score_file_name = score_file_path.split(".")[0]
    train_feature_df.to_csv(f"{feature_file_name}_train.csv", index=False)
    test_feature_df.to_csv(f"{feature_file_name}_test.csv", index=False)
    train_score_df.to_csv(f"{score_file_name}_train.csv", index=False)
    test_score_df.to_csv(f"{score_file_name}_test.csv", index=False)

    print(f"Number of training examples: {len(train_feature_df)}")
    print(f"Number of test examples: {len(test_feature_df)}")


if __name__ == "__main__":
    split("X_ordered_by_importance.csv", "y.csv")
