import pandas as pd
import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils import load_data

# Batch size, # input features, # output features
N, D_out = 64, 1
# Total number of iterations.
N_iter = 1000
# Dropout probability.
DROP_PROB = 0.5
INIT_LEARNING_RATE = 1e-4


def train(
        train_feature_df: pd.DataFrame,
        train_score_df: pd.DataFrame,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim,
        hidden_size: int,
        protease_type: str,
        writer: SummaryWriter,
):
    # [:, 1:] discards the target sequence names.
    X = torch.Tensor(train_feature_df.iloc[:, 1:].values)
    y = torch.Tensor(train_score_df.iloc[:, 1:].values)
    # Use 10% of training data as validation set.
    total_train = len(X)
    num_train = int(total_train * 0.9)
    X_train, X_valid = X[:num_train], X[num_train:]
    y_train, y_valid = y[:num_train], y[num_train:]
    min_valid_loss = float("+inf")

    for t in range(N_iter):
        pred_train = model(X_train)
        loss_train = loss_fn(pred_train, y_train)
        r2_score_train = metrics.r2_score(y_train.detach(), pred_train.detach())
        writer.add_scalar("R^2/train", r2_score_train, t)
        if t % 100 == 0:
            # Need to call `detach()` because r2_score calls `numpy()` on the tensors, but `numpy()` is not
            # callable on tensors in grad mode.
            print(
                f"Iteration #{t}. "
                f"Train loss: {loss_train}. "
                f"R^2 score: {r2_score_train}. "
            )

        # Perform validation and save the best model.
        with torch.no_grad():
            pred_valid = model(X_valid)
            loss_valid = loss_fn(pred_valid, y_valid)
            r2_score_valid = metrics.r2_score(y_valid, pred_valid)
            writer.add_scalar("R^2/valid", r2_score_valid, t)
            if loss_valid < min_valid_loss:
                min_valid_loss = loss_valid
                # Save the current best model to a file.
                torch.save(model, f"nn_{protease_type}_N_{N}_H_{hidden_size}.pt")

            if t % 100 == 0:
                print(
                    f"Iteration #{t}. "
                    f"Valid loss: {loss_valid}. "
                    f"R^2 score: {r2_score_valid}. "
                )

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()


def test(
        test_feature_df: pd.DataFrame,
        test_score_df: pd.DataFrame,
        test_output_save_path: str,
        model: nn.Module,
        loss_fn: nn.Module,
) -> float:
    X = torch.Tensor(test_feature_df.iloc[:, 1:].values)
    y = torch.Tensor(test_score_df.iloc[:, 1:].values)
    with torch.no_grad():
        pred_test = model(X)
        loss = loss_fn(pred_test, y)
        r2_score = metrics.r2_score(y, pred_test)
        print(f"Test loss: {loss}. R^2 score: {r2_score}")

        # Save the predicted scores to a csv file.
        preds_df = pd.DataFrame(pred_test).astype("float")
        column_labels = test_score_df.iloc[:, :1].reset_index(drop=True)
        preds_with_labels = pd.concat([column_labels, preds_df], axis=1)
        preds_with_labels.to_csv(test_output_save_path, index=False)

        return r2_score


def run(
        train_feature_df: pd.DataFrame,
        train_score_df: pd.DataFrame,
        test_feature_df: pd.DataFrame,
        test_score_df: pd.DataFrame,
        protease_type: str,
):
    # Number of input features = # of columns - 1. -1 because the first column is the label column.
    D_in = len(train_feature_df.iloc[0]) - 1

    r2_scores = []
    for hidden_size in [5, 10, 25, 50, 100, 200, 400, 800, 1600]:
        print(f"Protease type: {protease_type}. Training with hidden layer size {hidden_size}")
        # Simple two-layer neural network.
        model = nn.Sequential(
            nn.Linear(D_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
            nn.Linear(hidden_size, D_out),
        )
        # Using MSE loss since R^2 is just an affine transformation of MSE.
        loss_fn = nn.MSELoss(reduction="sum")
        optimizer = optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)

        # Setup Tensorboard.
        writer = SummaryWriter(log_dir=f"runs/nn_{protease_type}_N_{N}_H_{hidden_size}/")

        model.train()
        train(train_feature_df, train_score_df, model, loss_fn, optimizer, hidden_size, protease_type, writer)
        writer.flush()
        writer.close()

        # Load the best model and then run test.
        best_model = torch.load(f"nn_{protease_type}_N_{N}_H_{hidden_size}.pt")
        best_model.eval()
        test_r2_score = test(
            test_feature_df,
            test_score_df,
            f"nn_{protease_type}_N_{N}_H_{hidden_size}_preds.csv",
            best_model,
            loss_fn,
        )
        r2_scores.append(test_r2_score)

    print(f"All R^2 scores: {r2_scores}")


def main_furin():
    train_feature_df, train_score_df = load_data("X_ordered_by_importance_train.csv", "y_train.csv")
    test_feature_df, test_score_df = load_data("X_ordered_by_importance_test.csv", "y_test.csv")
    run(train_feature_df, train_score_df, test_feature_df, test_score_df, "furin")


def main_mmp(mmp_type: int):
    train_feature_df, train_score_df = load_data("data3_MMPs_X_train.csv", f"data3_MMP{mmp_type}_y_train.csv")
    test_feature_df, test_score_df = load_data("data3_MMPs_X_test.csv", f"data3_MMP{mmp_type}_y_test.csv")
    run(train_feature_df, train_score_df, test_feature_df, test_score_df, f"MMP{mmp_type}")


if __name__ == "__main__":
    main_furin()
    # for mmp_type in [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 24, 25]:
    #     main_mmp(mmp_type)
