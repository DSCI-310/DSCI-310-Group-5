import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src.mean_cross_val_scores import mean_cross_val_scores


class Data:

    def __init__(self):
        y_vals = np.random.choice([0, 1], size=40)
        dataset = pd.DataFrame({'x1': np.linspace(2,10,40), 'x2': np.linspace(4,10,40),
                   'x3': np.linspace(4,10,40), 'class': y_vals})
        train_df, test_df = train_test_split(dataset,
                                             test_size=0.3, random_state=123)
        self.X_train = train_df.drop(columns=["class"])
        self.X_test = test_df.drop(columns=["class"])
        self.y_train = train_df["class"]
        self.y_test = test_df["class"]


def test_mean_cross_val_correct_simple():
    """mean_cross_val_scores returns correct mean
    of cross validation with correct types of inputs""" 
    """test if it is a series"""
    """test if num of elements are as expected"""
    """test if elements bigger than 0 and smaller than 1"""
    dataset = Data()
    scale = StandardScaler()
    pipe_knn = make_pipeline(scale,
                             KNeighborsClassifier(n_neighbors=5))
    scoring = ['accuracy']
    result = mean_cross_val_scores(
        pipe_knn, dataset.X_train, dataset.y_train,
        return_train_score=True, scoring=scoring)
    assert all(result) <= 1
    assert all(result) >= 0
    assert result.count() == 4
    assert isinstance(result, pd.core.series.Series)


def test_mean_cross_val_correct():
    """mean_cross_val_scores returns correct mean
    of cross validation with correct types of inputs""" 
    """test if it is a series"""
    """test if num of elements are as expected"""
    """test if elements bigger than 0 and smaller than 1"""
    dataset = Data()
    scale = StandardScaler()
    pipe_knn = make_pipeline(scale,
                             DecisionTreeClassifier(random_state=123))
    scoring = ['accuracy', "f1", "recall", "precision"]
    result = mean_cross_val_scores(
        pipe_knn, dataset.X_train, dataset.y_train,
        return_train_score=True, scoring=scoring)
    assert all(result) <= 1
    assert all(result) >= 0
    assert result.count() == 10
    assert isinstance(result, pd.core.series.Series)
