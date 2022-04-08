import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)


def plot_cm(model, X_train, y_train, X_test, y_test, title):
    """
    Returns confusion matrix on predictions of y_test with given title 
    of given model fitted X_train and y_train 
    -----------
    PARAMETERS:
    model :
        scikit-learn model or sklearn.pipeline.Pipeline
    X_train : numpy array or pandas DataFrame/Series
        X in the training data
    y_train : numpy array or pandas DataFrame/Series
        y in the training data
    X_test : numpy array or pandas DataFrame/Series
        X in the testing data
    y_test : numpy array or pandas DataFrame/Series
        y in the testing data
    -----------
    REQUISITES:
    X_train, y_train, X_test, y_test cannot be empty.
    -----------
    RETURNS:
    A sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object 
    -----------
    Examples

    plot_cm(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, "Fig")
    """
    if not isinstance(X_train, (pd.core.series.Series,
                                pd.core.frame.DataFrame, np.ndarray)):
        raise TypeError("'X_train' should be of type numpy.array or pandas.Dataframe")
    if not isinstance(y_train, (pd.core.series.Series,
                                pd.core.frame.DataFrame, np.ndarray)):
        raise TypeError("'y_train' should be of type numpy.array or pandas.Dataframe")
    if not isinstance(X_test, (pd.core.series.Series,
                               pd.core.frame.DataFrame, np.ndarray)):
        raise TypeError("'X_test' should be of type numpy.array or pandas.Dataframe")
    if not isinstance(y_test, (pd.core.series.Series,
                               pd.core.frame.DataFrame, np.ndarray)):
        raise TypeError("'y_test' should be of type numpy.array or pandas.Dataframe")
    if not isinstance(title, str):
        raise TypeError("'title' should be of 'str'")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.title(title)
    return disp