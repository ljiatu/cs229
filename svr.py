from typing import Sequence

import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from utils import load_data

KERNEL_TYPES = ["linear", "poly", "rbf", "sigmoid"]
REGULARIZATION_STRENGTHS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MODEL_SAVE_PATH_FORMAT = "svr_model_k_{}_c_{}.joblib"
BEST_MODEL_SAVE_PATH = "svr_model_best.joblib"


def train(
        train_feature_df: pd.DataFrame,
        train_score_df: pd.DataFrame,
        kernel_types: Sequence[str],
        reg_strengths: Sequence[float],
        best_model_save_path: str,
) -> SVR:
    """
    Train the model with the given inputs and returns the trained model.
    """
    assert len(reg_strengths) > 0
    # Drop the first column, which contains labels for target sequences.
    X = train_feature_df.to_numpy()[:, 1:]
    # y initially comes out as a column vector. Use ravel() to make it a row vector instead.
    y = train_score_df.to_numpy()[:, 1:].ravel()

    best_clf = None
    best_val_score = float("-inf")
    for k in kernel_types:
        for c in reg_strengths:
            clf = SVR(kernel=k, C=c)
            clf.fit(X, y)
            # Split the training data into 5 folds and perform cross validation.
            val_scores = cross_val_score(clf, X, y, cv=5)
            val_score = val_scores.mean()
            print(f"Kernel type: {k}. Reg strength {c}. Mean val score: {val_score:0.2f} (+/- {val_scores.std() * 2:0.2f})")
            # Save the individual models.
            dump(clf, MODEL_SAVE_PATH_FORMAT.format(k, c))
            if val_score > best_val_score:
                best_clf = clf
                best_val_score = val_score

    # Save the best model to a file.
    dump(best_clf, best_model_save_path)
    return best_clf


def test(test_feature_df: pd.DataFrame, test_score_df: pd.DataFrame, clf: SVR, test_output_save_path: str) -> None:
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
    print(f"Test score: {test_score}")


def main() -> None:
    train_feature_df, train_score_df = load_data("X_ordered_by_importance_train.csv", "y_train.csv")
    test_feature_df, test_score_df = load_data("X_ordered_by_importance_test.csv", "y_test.csv")
    clf = train(train_feature_df, train_score_df, KERNEL_TYPES, REGULARIZATION_STRENGTHS, BEST_MODEL_SAVE_PATH)
    # clf = load("svr_model_best.joblib")
    test(test_feature_df, test_score_df, clf, "svr_preds.csv")


if __name__ == "__main__":
    main()
