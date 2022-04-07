import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.plot_confusion_matrix import plot_cm


col_names = ["id", "clump", "unif_size", "unif_shape", "adhesion", "epi_size",
             "nuclei", "chromatin", "nucleoli", "mitoses", "class"]

dataset = pd.read_csv("../data/raw/breast_cancer.txt", names=col_names, sep=",")
df = dataset.iloc[:50]
df = df.loc[:, ['clump', 'unif_size', 'unif_shape', 'class']]
df['class'] = df['class'].replace([2],0)
df['class'] = df['class'].replace([4],1) 
df = df[(df != '?').all(axis=1)]
train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)
X_train = train_df.drop(columns=["class"])
X_test = test_df.drop(columns=["class"])
y_train = train_df["class"]
y_test = test_df["class"]
pipe_reg = make_pipeline(StandardScaler(), LogisticRegression())


def test_cm_two_classes_target():
    """
    Test confusion matrix readability (labels), return type, 
    and number of return values for two classes target 
    (y_train and y_test have 2 unique classes) 
    """
    plot = plot_cm(pipe_reg, X_train, y_train, X_test, y_test, "Fig 3")
    assert plot.text_.shape == (2, 2)
    assert plot.ax_.get_xlabel() == 'Predicted label'
    assert plot.ax_.get_ylabel() == 'True label'
    assert isinstance(plot, sklearn.metrics._plot.confusion_matrix.
                      ConfusionMatrixDisplay)


def test_cm_three_classes_target():
    """
    Test confusion matrix readability (labels), return type, 
    and number of return values for three classes target 
    (y_train or y_test have three unique classes) 
    """
    test_3 =  pd.DataFrame({'clump':[4],'unif_size':[3],'unif_shape':[1],'class':[2]})
    X_train_3 = pd.concat([X_train, test_3.iloc[:,:3]])
    y_train_3 = pd.concat([y_train, test_3.iloc[:, 3]])
    plot3 = plot_cm(pipe_reg, X_train_3, y_train_3, X_test, y_test, "Confusion Matrix")
    assert plot3.text_.shape == (3, 3)
    assert plot3.ax_.get_xlabel() == 'Predicted label'
    assert plot3.ax_.get_ylabel() == 'True label'
    assert isinstance(plot3, sklearn.metrics._plot.confusion_matrix.
                      ConfusionMatrixDisplay)
