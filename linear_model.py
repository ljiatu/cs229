import multiprocessing;
import numpy as np;
import pandas as pd;
from joblib import dump;
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso;


from utils import load_data

alphas = 10**np.linspace(-2, 10, 100)* 0.5;
MODEL_SAVE_PATH_FORMAT = "svr_model_k_{}_c_{}.joblib"
BEST_RIDGE_MODEL_SAVE_PATH = "Ridge_model_best.joblib"
BEST_LASSO_MODEL_SAVE_PATH = "Lasso_model_best.joblib"


def train_ridgeCV(train_feature_df: pd.DataFrame, train_score_df: pd.DataFrame, alphas: np.array) -> tuple:
    """
    Train the model with the given inputs and returns the trained model.
    
    ----------
    Parameters
    ----------
    train_feature_df : pd.DataFrame:
        The design matrix X in the lecture.
        
    train_score_df : pd.DataFrame:
        The output y in the lecture.
        
    alphas : np.array:
        The tuning hyperparameters.
        
    -------
    Returns
    -------
    tuple(float, Ridge):
        1st: The R2 score.
        2nd: The Ridge Model.

    """
    print("Training Ridge linear model with 5-fold cross-validation");
    # Drop the first column, which contains labels for target sequences.
    X = train_feature_df.to_numpy()[:, 1:];
    # y initially comes out as a column vector. Use ravel() to make it a row vector instead.
    y = train_score_df.to_numpy()[:, 1:].ravel();

    ridgecv = RidgeCV(alphas, normalize = True, cv = 5);
    # clf.fit(X, y)
    # Split the training data into 5 folds and perform cross validation.
    ridgecv = ridgecv.fit(X, y);
    val_score = ridgecv.score(X, y);
    # Save the individual models.
    #dump(clf, MODEL_SAVE_PATH_FORMAT.format(kernel_type, reg_strength))
    return val_score, ridgecv;


def test_ridgeCV(test_feature_df: pd.DataFrame, test_score_df: pd.DataFrame, ridgecv: RidgeCV, test_output_save_path: str) -> None:
    """
    Predicts on test data, and saves the test output on test_output_save_path.

    ----------
    Parameters
    ----------
    test_feature_df : pd.DataFrame:
        The test X.
        
    test_score_df : pd.DataFrame:
        The test y.
        
    clf : RidgeCV:
        The trained RidgeCV linear regression model.
        
    test_output_save_path : str:
        The output file name.

    -------
    Returns
    -------
    None

    """
    X = test_feature_df.to_numpy()[:, 1:];
    y = test_score_df.to_numpy()[:, 1:].ravel();

    preds_df = pd.DataFrame(ridgecv.predict(X));
    column_labels = test_score_df.iloc[:, :1].reset_index(drop=True);
    preds_with_labels = pd.concat([column_labels, preds_df], axis=1);
    preds_with_labels.to_csv(test_output_save_path, index=False);

    test_score = ridgecv.score(X, y);
    print(f"Ridge regression test score: {test_score}")
    
def train_lassoCV(train_feature_df: pd.DataFrame, train_score_df: pd.DataFrame) -> tuple:
    """
    Train the model with the given inputs and returns the trained model.
    
    ----------
    Parameters
    ----------
    train_feature_df : pd.DataFrame:
        The design matrix X in the lecture.
        
    train_score_df : pd.DataFrame:
        The output y in the lecture.
        
    -------
    Returns
    -------
    tuple(float, LassoCV):
        1st: The R2 score.
        2nd: The Lasso Model.

    """
    print("Training Lasso linear model with 5-fold cross-validation");
    # Drop the first column, which contains labels for target sequences.
    X = train_feature_df.to_numpy()[:, 1:];
    # y initially comes out as a column vector. Use ravel() to make it a row vector instead.
    y = train_score_df.to_numpy()[:, 1:].ravel();

    lassocv = LassoCV(max_iter = 100000, normalize = True, cv = 5);
    # clf.fit(X, y)
    # Split the training data into 5 folds and perform cross validation.
    lassocv = lassocv.fit(X, y);
    val_score = lassocv.score(X, y);
    # Save the individual models.
    #dump(clf, MODEL_SAVE_PATH_FORMAT.format(kernel_type, reg_strength))
    return val_score, lassocv;


def test_lassoCV(test_feature_df: pd.DataFrame, test_score_df: pd.DataFrame, lassocv: LassoCV, test_output_save_path: str) -> None:
    """
    Predicts on test data, and saves the test output on test_output_save_path.

    ----------
    Parameters
    ----------
    test_feature_df : pd.DataFrame:
        The test X.
        
    test_score_df : pd.DataFrame:
        The test y.
        
    clf : LassoCV:
        The trained Lasso linear regression model.
        
    test_output_save_path : str:
        The output file name.

    -------
    Returns
    -------
    None

    """
    X = test_feature_df.to_numpy()[:, 1:];
    y = test_score_df.to_numpy()[:, 1:].ravel();

    preds_df = pd.DataFrame(lassocv.predict(X));
    column_labels = test_score_df.iloc[:, :1].reset_index(drop=True);
    preds_with_labels = pd.concat([column_labels, preds_df], axis=1);
    preds_with_labels.to_csv(test_output_save_path, index=False);

    test_score = lassocv.score(X, y);
    print(f"Lasso regression test score: {test_score}")
    
if __name__ == "__main__":
    train_feature_df, train_score_df = load_data("X_ordered_by_importance_train.csv", "y_train.csv");
    test_feature_df, test_score_df = load_data("X_ordered_by_importance_test.csv", "y_test.csv");
    #pool = multiprocessing.Pool(processes=5);
    #train_args = [(train_feature_df, train_score_df, k, c) for k in KERNEL_TYPES for c in REGULARIZATION_STRENGTHS]
    #rets = pool.starmap(train, train_args)

    best_score_ridgecv, best_ridgecv = train_ridgeCV(train_feature_df, train_score_df, alphas);
    print(f"Best alpha for Ridge regression: {best_ridgecv.alpha_} with score: {best_score_ridgecv}");
    dump(best_ridgecv, BEST_RIDGE_MODEL_SAVE_PATH);
    # clf = load("svr_model_best.joblib")
    test_ridgeCV(test_feature_df, test_score_df, best_ridgecv, "Ridge_preds.csv")
    
    best_score_lassocv, best_lassocv = train_lassoCV(train_feature_df, train_score_df);
    print(f"Best alpha for Lasso regression: {best_lassocv.alpha_} with score: {best_score_lassocv}");
    dump(best_ridgecv, BEST_RIDGE_MODEL_SAVE_PATH);
    # clf = load("svr_model_best.joblib")
    test_lassoCV(test_feature_df, test_score_df, best_lassocv, "Lasso_preds.csv")
    dump(best_lassocv, BEST_LASSO_MODEL_SAVE_PATH);