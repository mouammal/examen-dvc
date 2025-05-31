import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def import_dataset(file_path):
    print(f"Importing dataset from {file_path}")
    return pd.read_csv(file_path)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    print(f"Data split into training and testing sets: {len(X_train)} training samples and {len(X_test)} testing samples")
    return X_train, X_test, y_train, y_test


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
        print(f"Saved {filename} to {output_filepath}")

# Import datasets
file_path = "./data/raw_data/raw.csv"
df = import_dataset(file_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(df)

# Save dataframes to their respective output file paths
output_filepath =  "./data/processed_data"
save_dataframes(X_train, X_test, y_train, y_test, output_filepath)