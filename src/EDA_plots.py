import pandas as pd
import numpy as np
import argparse
from plot_hist import plot_hist_overlay
from plot_boxplot import boxplot_plotting
import matplotlib.pyplot as plt


def EDA_plot(train_df, hist_output, boxplot_output):
    train_df = pd.read_csv(train_df)
    X_train = train_df.drop(columns=["class"])
    numeric_looking_columns = X_train.select_dtypes(
        include=np.number).columns.tolist()
    benign_cases = train_df[train_df["class"] == 0]
    malignant_cases = train_df[train_df["class"] == 1]
    #plot histogram
    for idx, x in enumerate(numeric_looking_columns): 
        plot_hist_overlay(benign_cases, malignant_cases, 
                          x, labels=["0 - benign", "1 - malignant"],
                          fig_no=f"1.{idx}")
        plt.savefig(hist_output)
    #plot boxplot 
    boxplot_plotting(3, 3, 20, 25, numeric_looking_columns, train_df, 2)
    plt.savefig(boxplot_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plots EDA")
    parser.add_argument("train_df", type=str, help="Path to train_df")
    parser.add_argument("hist_output", type=str, help="Path to histogram output")
    parser.add_argument("boxplot_output", type=str, help="Path to boxplot output")
    args = parser.parse_args()
    EDA_plot(args.train_df, args.hist_output, args.boxplot_output)