import pandas as pd


def split(file_path: str) -> None:
    df = pd.read_csv(file_path)
    labels_df = df.iloc[:, 0]
    mmp_types = list(df.columns)
    for i in range(1, len(df.columns)):
        scores_df = pd.concat([labels_df, df.iloc[:, i]], axis=1)
        scores_df.to_csv(f"data3_{mmp_types[i]}_y.csv", index=False)


if __name__ == "__main__":
    split("data3_MMPs_y.csv")
