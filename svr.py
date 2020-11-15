import multiprocessing
from typing import Tuple

import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from utils import load_data

KERNEL_TYPES = ["rbf"]
REGULARIZATION_STRENGTHS = [1, 2, 4, 8, 16, 32]


def train(
        train_feature_df: pd.DataFrame,
        train_score_df: pd.DataFrame,
        kernel_type: str,
        reg_strength: int,
        mmp_type: int,
) -> Tuple[float, SVR, int]:
    """
    Train the model with the given inputs and returns the trained model.
    """
    print(f"MMP type {mmp_type}. Training SVR model with kernel type {kernel_type} and regularization strength {reg_strength}")
    # Drop the first column, which contains labels for target sequences.
    X = train_feature_df.to_numpy()[:, 1:]
    # y initially comes out as a column vector. Use ravel() to make it a row vector instead.
    y = train_score_df.to_numpy()[:, 1:].ravel()

    clf = SVR(kernel=kernel_type, C=reg_strength)
    clf.fit(X, y)
    # Split the training data into 5 folds and perform cross validation.
    val_scores = cross_val_score(clf, X, y, cv=5)
    val_score = val_scores.mean()
    print(
        f"Kernel type: {kernel_type}. Reg strength {reg_strength}. "
        f"Mean val score: {val_score:0.2f} (+/- {val_scores.std() * 2:0.2f})"
    )
    # Save the individual models.
    dump(clf, f"svr_model_MMP{mmp_type}_k_{kernel_type}_c_{reg_strength}.joblib")
    return val_score, clf, reg_strength


def test(
        test_feature_df: pd.DataFrame,
        test_score_df: pd.DataFrame,
        clf: SVR,
        test_output_save_path: str,
        mmp_type: int,
        reg_strength: int,
) -> None:
    """
    Predicts on test data, and saves the test output on test_output_save_path.
    """
    X = test_feature_df.to_numpy()[:, 1:]
    y = test_score_df.to_numpy()[:, 1:].ravel()

    preds_df = pd.DataFrame(clf.predict(X))
    column_labels = test_score_df.iloc[:, :1].reset_index(drop=True)
    preds_with_labels = pd.concat([column_labels, preds_df], axis=1)
    preds_with_labels.to_csv(test_output_save_path, index=False)

    test_score = clf.score(X, y)
    print(f"MMP type: {mmp_type}. Reg strength: {reg_strength}. Test score: {test_score}")


def main(mmp_type: int) -> None:
    train_feature_df, train_score_df = load_data("data3_MMPs_X_train.csv", f"data3_MMP{mmp_type}_y_train.csv")
    test_feature_df, test_score_df = load_data("data3_MMPs_X_test.csv", f"data3_MMP{mmp_type}_y_test.csv")

    pool = multiprocessing.Pool(processes=6)
    train_args = [
        (train_feature_df.copy(), train_score_df.copy(), k, c)
        for k in KERNEL_TYPES for c in REGULARIZATION_STRENGTHS
    ]
    rets = pool.starmap(train, train_args)

    best_clf = None
    # best_val_score = float("-inf")
    for val_score, clf, reg_strength in rets:
        test(test_feature_df, test_score_df, clf, f"svr_MMP{mmp_type}_preds.csv", mmp_type, reg_strength)
        # if val_score > best_val_score:
        # best_val_score = val_score
        # best_clf = clf

    # dump(best_clf, f"svr_model_MMP{mmp_type}_best.joblib")


if __name__ == "__main__":
    for mmp_type in [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 24, 25]:
        main(mmp_type)
