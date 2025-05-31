import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def import_dataset(file_path):
    print(f"Importing dataset from {file_path}")
    return pd.read_csv(file_path)

def scale_datasets(X_train, X_test):
    # Séparer les colonnes à exclure
    X_train_features = X_train.drop(columns=['date'])
    X_test_features = X_test.drop(columns=['date'])

    # Appliquer le scaler
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train_features)
    X_test_scaled_array = scaler.transform(X_test_features)

    # Convertir en DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train_features.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test_features.columns, index=X_test.index)
    print(f"Datasets scaled: {len(X_train_scaled)} training samples and {len(X_test_scaled)} testing samples")

    return X_train_scaled, X_test_scaled

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
        print(f"Saved {filename} to {output_filepath}")


# Import datasets Train et Test 
train_path = "./data/processed_data/X_train.csv"
test_path = "./data/processed_data/X_test.csv"
X_train = import_dataset(train_path)
X_test = import_dataset(test_path)

# Scale the datasets
X_train_scaled, X_test_scaled = scale_datasets(X_train, X_test)

# Save dataframes to their respective output file paths
output_filepath =  "./data/processed_data"
save_dataframes(X_train_scaled, X_test_scaled, output_filepath)