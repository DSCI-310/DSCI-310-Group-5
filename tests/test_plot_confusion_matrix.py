import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.plot_confusion_matrix import plot_cm


y_vals = np.random.choice([0, 1], size=40)
df = pd.DataFrame({'x1': np.linspace(2,10,40), 'x2': np.linspace(4,10,40),
                   'x3': np.linspace(4,10,40), 'class': y_vals})
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
    test_3 =  pd.DataFrame({'x1': [4], 'x2' : [3], 'x3' : [1], 'class' : [2]})
    X_train_3 = pd.concat([X_train, test_3.iloc[:,:3]])
    y_train_3 = pd.concat([y_train, test_3.iloc[:, 3]])
    plot3 = plot_cm(pipe_reg, X_train_3, y_train_3, X_test, y_test, "Confusion Matrix")
    assert plot3.text_.shape == (3, 3)
    assert plot3.ax_.get_xlabel() == 'Predicted label'
    assert plot3.ax_.get_ylabel() == 'True label'
    assert isinstance(plot3, sklearn.metrics._plot.confusion_matrix.
                      ConfusionMatrixDisplay)
