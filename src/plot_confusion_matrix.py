import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def plot_cm(model, X_train, y_train, X_test, y_test, title):
    """
    Returns confusion matrix on predictions of y_test with given title 
    of given model fitted X_train and y_train 
    -----------
    Parameters
    -----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array or pandas DataFrame
        y in the training data
    X_test : numpy array or pandas DataFrame
        X in the testing data
    y_test : numpy array or pandas DataFrame
        y in the testing data
    Returns
    -----------
        A sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object 
    """

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.title(title)
    return disp