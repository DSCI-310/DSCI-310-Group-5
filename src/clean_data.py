from sklearn.model_selection import train_test_split
import pandas as pd
import argparse


def clean_data (input_path, output_path_train, output_path_test):
	"""The current function cleans up the data of any missing values. It also re-groups the data into only two target classes rather than 4. Further on the data is split into training and testing data."""
    #cleaning data
	df = pd.read_csv(str(input_path))
	df = df[(df != '?').all(axis=1)]
	df['nuclei'] = df['nuclei'].astype(int)
	df = df.drop(columns=["id"])
	#replace 2 -> 0 & 4 -> 1 in target class 
	df['class'] = df['class'].replace([2],0)
	df['class'] = df['class'].replace([4],1) 
	#split train/test data
	train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)
	train_df.to_csv(str(output_path_train), index=False)
	test_df.to_csv(str(output_path_test), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean data")
    parser.add_argument("input_path", help="Path to dataset")
    parser.add_argument("output_path_train", help="Path to train output")
    parser.add_argument("output_path_test", help="Path to test output")
    args = parser.parse_args()
    clean_data(args.input_path, args.output_path_train, args.output_path_test)