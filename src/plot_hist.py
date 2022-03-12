import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_hist_overlay(df0, df1, column, labels, fig_no="1",alpha=0.7, bins=5, **kwargs):
    """
    A function that plot histogram for a target
    classification label against each numerical features.

    REQUIRED: target label are binary i.e 0 or 1, negative or positive

    Parameters
    -------
    df0:
        A pandas DataFrame that is corresponded to the label 0
    df1:
        A pandas DataFrame that is corresponded to the label 1
    columns:
        A column name
    labels: 
        A list of label for each of the histograms for each label
    fig_no: optional, default="1"
        A string denoting the figure number, in the case of multiple figures
    alpha: optional, default=0.7
        A float denotes the alpha value for the matplotlib hist function
    bin: optional, default=5
        An int denotes the number of bins for the matplotlib hist function
    **kwargs:
        Other parameters for the plotting function 

    Returns
    -------
    A matplotlib.axes.Axes object

    Examples
    -------
    benign_cases = train_df[train_df["class"] == 0]   # df0             
    malignant_cases = train_df[train_df["class"] == 1] # df1

    plot_hist_overlay(benign_cases, malignant_cases,"unif_size", labels=["0 - benign", "1 - malignant"]
    
    """
    column_name = column.title().replace("_", " ")
    fig, ax = plt.subplots()
    ax.hist(df0[column], alpha=alpha, bins=bins, label=labels[0], **kwargs, figure=fig)
    ax.hist(df1[column], alpha=alpha, bins=bins, label=labels[1], **kwargs, figure=fig)
    ax.legend(loc="upper right")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Figure {fig_no}: Histogram of {column_name} for each target class label")
    return ax
