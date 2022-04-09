import argparse
import pandas as pd


def load_data (input_path, output_path):
	"""The objective with this function is to load the data from a csv file and set the column names to create a proper pandas dataframe. We have to give the names to the columns of the dataframe."""
	col_names = ["id", "clump", "unif_size", "unif_shape", "adhesion",
				 "epi_size", "nuclei", "chromatin", "nucleoli",
				 "mitoses", "class"]
	dataset = pd.read_csv(str(input_path), names=col_names, sep=",")
	return dataset.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load data")
    parser.add_argument("input_path", help="Path to data source")
    parser.add_argument("output_path", help="Path to output")
    args = parser.parse_args()
    load_data(args.input_path, args.output_path)
