import multiprocessing;
import numpy as np;
import pandas as pd;
from joblib import dump;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern;
from sklearn.model_selection import cross_val_score;

from utils import load_data

KERNEL_TYPES = [None, "RationalQuadratic()", "Matern()"]; # None is 1.0 * RBF(1.0)
MODEL_SAVE_PATH_FORMAT = "svr_model_k_{}_c_{}.joblib"
BEST_RBF_MODEL_SAVE_PATH = "RBF_model_best.joblib"
BEST_RationalQuadratic_MODEL_SAVE_PATH = "RationalQuadratic_model_best.joblib"
BEST_Matern_MODEL_SAVE_PATH = "Matern_model_best.joblib"
BEST_GP_MODEL_SAVE_PATH = "GP_model_best.joblib"


def train_GP(train_feature_df: pd.DataFrame, train_score_df: pd.DataFrame, kernel: str) -> tuple:
    """
    Train the model with the given inputs and returns the trained model.
    
    ----------
    Parameters
    ----------
    train_feature_df : pd.DataFrame:
        The design matrix X in the lecture.
        
    train_score_df : pd.DataFrame:
        The output y in the lecture.
        
    kernel : str:
        The kernel type.
        
    -------
    Returns
    -------
    tuple(float, GP):
        1st: The R2 score.
        2nd: The GP Model.

    """
    print("Training Gaussian Process regressor");
    # Drop the first column, which contains labels for target sequences.
    X = train_feature_df.to_numpy()[:, 1:];
    # y initially comes out as a column vector. Use ravel() to make it a row vector instead.
    y = train_score_df.to_numpy()[:, 1:].ravel();
    
    best_GP = None;
    best_score = float("-inf");
    for k in kernel:
        GP = GaussianProcessRegressor(k).fit(X, y);
        val_score = GP.score(X, y);
        print(k, val_score);
        if (val_score > best_score):
            best_GP = GP;
            best_score = val_score;
    
    return best_score, GP;


def test_GP(test_feature_df: pd.DataFrame, test_score_df: pd.DataFrame, GP, test_output_save_path: str) -> None:
    """
    Predicts on test data, and saves the test output on test_output_save_path.

    ----------
    Parameters
    ----------
    test_feature_df : pd.DataFrame:
        The test X.
        
    test_score_df : pd.DataFrame:
        The test y.
        
    GP:
        The trained GP model.
        
    test_output_save_path : str:
        The output file name.

    -------
    Returns
    -------
    None

    """
    X = test_feature_df.to_numpy()[:, 1:];
    y = test_score_df.to_numpy()[:, 1:].ravel();

    preds_df = pd.DataFrame(GP.predict(X));
    column_labels = test_score_df.iloc[:, :1].reset_index(drop=True);
    preds_with_labels = pd.concat([column_labels, preds_df], axis=1);
    preds_with_labels.to_csv(test_output_save_path, index=False);

    test_score = GP.score(X, y);
    print(f"GP test score: {test_score}");
    
if __name__ == "__main__":
    train_feature_df, train_score_df = load_data("X_ordered_by_importance_train.csv", "y_train.csv");
    test_feature_df, test_score_df = load_data("X_ordered_by_importance_test.csv", "y_test.csv");
    #pool = multiprocessing.Pool(processes=5);
    #train_args = [(train_feature_df, train_score_df, k, c) for k in KERNEL_TYPES for c in REGULARIZATION_STRENGTHS]
    #rets = pool.starmap(train, train_args)

    best_score_GP, best_GP = train_GP(train_feature_df, train_score_df, KERNEL_TYPES);
    print(f"Best kernel for GP: {best_GP.kernel_} with score: {best_score_GP}");
    dump(best_GP, BEST_GP_MODEL_SAVE_PATH);
    # clf = load("svr_model_best.joblib")
    test_GP(test_feature_df, test_score_df, best_GP, "GP_preds.csv");