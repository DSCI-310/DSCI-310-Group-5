import pandas as pd
from sklearn.model_selection import cross_validate


def mean_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean of cross validation of given model, X_train, y_train 

    Parameters
    -----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array or pandas DataFrame
        y in the training data

    Returns
    -----------
        pandas Series with mean scores from cross_validation
    -----------
    Inspired by mean_std_cross_val_scores in UBC CPSC330 Lecture 8 term Winter 2021
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)
    mean_scores = pd.DataFrame(scores).mean()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append(mean_scores[i])

    return pd.Series(data=out_col, index=mean_scores.index)