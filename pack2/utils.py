import numpy as np
import pandas as pd
import seaborn
from sklearn.impute import SimpleImputer


def correl(df_in, threshold):
    df_corr = df_in.corr().abs()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    tri_df = df_corr.mask(mask)
    seaborn.heatmap(tri_df, cmap="coolwarm", annot=True)
    # to_drop = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
    # reduced_df = df_in.drop(to_drop, axis=1)
    # print(f'dropped: {to_drop}')


def fill_missing_values(dataset):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(dataset)
    imp.transform(dataset)


def preprocess_dataset(dataset) -> pd.DataFrame:
    dataset = dataset.copy()
    fill_missing_values(dataset)
    correl(dataset, 0.6)
    return dataset


def get_dataset():
    dataset_white = pd.read_csv("data/winequality-white.csv", delimiter=";")
    dataset_red = pd.read_csv("data/winequality-red.csv", delimiter=";")
    dataset_white["type"] = 0
    dataset_red["type"] = 1
    total = pd.concat([dataset_white, dataset_red])
    return preprocess_dataset(dataset=total)
