from sklearn.model_selection import train_test_split
import pandas as pd
import argparse


def clean_data(dataset, train_df, test_df):
    #cleaning data
    df = pd.read_csv(dataset)
    df = df[(df != '?').all(axis=1)]
    df['nuclei'] = df['nuclei'].astype(int)
    df = df.drop(columns=["id"])
    #replace 2 -> 0 & 4 -> 1 in target class 
    df['class'] = df['class'].replace([2],0)
    df['class'] = df['class'].replace([4],1) 
    #split train/test data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)
    train_df.to_csv(train_df) #changed args
    test_df.to_csv(test_df) #changed arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean data")
    parser.add_argument(dataset, type=str, help="Path to dataset") #changed
    parser.add_argument(train_df, type=str, help="Path to dataset") #changed
    parser.add_argument(test_df, type=str, help="Path to dataset") #changed
    args = parser.parse_args()
    clean_data(args.dataset, args.train_df, args.test_df)
