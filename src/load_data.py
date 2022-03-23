import argparse
import pandas as pd


def load_data(path):
    col_names = ["id", "clump", "unif_size", "unif_shape", "adhesion",
                 "epi_size", "nuclei", "chromatin", "nucleoli",
                 "mitoses", "class"]
    dataset = pd.read_csv(path, names=col_names, sep=",")
    return dataset.to_csv('../data/raw/breast_cancer.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load data")
    parser.add_argument("path", type=str, help="Path to data source")
    args = parser.parse_args()
    load_data(args.path)
