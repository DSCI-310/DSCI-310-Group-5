import mean_cross_val
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class Data:

    def __init__(self):
        col_name = ['brand.name', 'tar.content', 
                    'nicotine.content', 'weight', 'carbon.monoxide.content']
        ds = pd.read_csv("cigarettes_c.txt", names=col_name, sep=",")
        ds = ds.drop(columns=["brand.name"])
        train_df, test_df = train_test_split(ds, test_size=0.3, random_state=123)
        self.X_train = train_df.drop(columns=["carbon.monoxide.content"])
        self.X_test = test_df.drop(columns=["carbon.monoxide.content"])
        self.y_train = train_df["carbon.monoxide.content"]
        self.y_test = test_df["carbon.monoxide.content"]


def test_mean_cross_val_correct():
    """mean_cross_val_scores returns correct mean
    of cross validation with correct types of inputs""" 
    """test if it is a series"""
    """test if num of elements are the same"""
    """test if elements bigger than 0 and smaller than 1"""
    dataset = Data()
    scale = StandardScaler()
    pipe_knn = make_pipeline(scale,
                             KNeighborsClassifier(n_neighbors=5))
    scoring = ['accuracy']
    result = mean_cross_val.mean_cross_val_scores(
        pipe_knn, dataset.X_train, dataset.y_train,
        return_train_score=True, scoring=scoring)
    assert result[1] < 1
    assert result[1] >= 0
    assert isinstance(result, series)

