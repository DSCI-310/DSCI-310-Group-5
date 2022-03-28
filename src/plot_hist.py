import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_hist_overlay(df0, df1, columns, labels, fig_no="1",alpha=0.7, bins=5, **kwargs):
    """
    A function that plot multiple histogram for a target
    classification label against each numerical features.
    The resulting histograms will be a grid layout contained in
    one single Figure object

    REQUIRED: target label are binary i.e 0 or 1, negative or positive
    Parameters
    -------
    df0:
        A pandas DataFrame that is corresponded to the label 0
    df1:
        A pandas DataFrame that is corresponded to the label 1
    columns:
        A list of column name
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
    A matplotlib.figure.Figure object

    Examples
    -------
    benign_cases = train_df[train_df["class"] == 0]   # df0             
    malignant_cases = train_df[train_df["class"] == 1] # df1

    plot_hist_overlay(benign_cases, malignant_cases,["unif_size"], labels=["0 - benign", "1 - malignant"]
    
    """
    # These are legacy codes are comment out in case we need to reuse in the future
    # column_name = column.title().replace("_", " ")
    # fig, ax = plt.subplots()
    # ax.hist(df0[column], alpha=alpha, bins=bins, label=labels[0], **kwargs, figure=fig)
    # ax.hist(df1[column], alpha=alpha, bins=bins, label=labels[1], **kwargs, figure=fig)
    # ax.legend(loc="upper right")
    # ax.set_xlabel(column_name)
    # ax.set_ylabel("Count")
    # ax.set_title(f"Figure {fig_no}: Histogram of {column_name} for each target class label")
    # return ax


    # To automatically calculating the size of dimension of the figures (Square shape)
    size = len(columns)
    dim = np.ceil(np.sqrt([size])).astype(int)[0]
    fig = plt.figure(1, figsize=(22,22))

    for idx, x in enumerate(columns):
        pos = (dim * 100 + dim * 10) + (idx + 1)
        subplot=plt.subplot(pos)
        col_name = x.title().replace("_", " ")
        subplot.hist(df0[x], alpha=alpha, bins=bins, label=labels[0], **kwargs)
        subplot.hist(df1[x], alpha=alpha, bins=bins, label=labels[1], **kwargs)
        subplot.legend(loc="upper right")
        subplot.set_xlabel(col_name, fontsize=14)
        subplot.set_ylabel("Count", fontsize=14)
        subplot.set_title(f"Figure {fig_no}.{idx+1}: Histogram of {col_name} for each target class label", 
                          fontsize=14)
    fig.suptitle(f"Figure {fig_no}: Distribution of the target class for each numeric feature", fontsize=20)

    return fig
