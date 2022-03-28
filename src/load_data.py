import argparse
import pandas as pd

## Load data from data directory and adds column names. Remove "col_names" code if data has column names

def load_data(src_path, dest_path):
    col_names = ["id", "clump", "unif_size", "unif_shape", "adhesion",
                 "epi_size", "nuclei", "chromatin", "nucleoli",
                 "mitoses", "class"]
    dataset = pd.read_csv(src_path, names=col_names, sep=",")
    dataset.to_csv(dest_path) #deleted return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load data")
    parser.add_argument("src_path", type=str, help="Path to data source")
    parser.add_argument("dest_path", type=str, help="Path to data source") #need two arguments # added second arg
    args = parser.parse_args()
    load_data(args.src_path, args.dest_path)
