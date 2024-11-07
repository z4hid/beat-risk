import pandas as pd
import sys
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load parameters from YAML file
params = yaml.safe_load(open('params.yaml'))['preprocess']


# Data Preprocess
def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)
    # check missing values
    if data.isnull().values.any():
        print("Data contains missing values")
    else:
        print("Data does not contain missing values")
    
    categorical_columns = data.select_dtypes(include=['object']).columns.to_list()
    # numerical_columns = data.select_dtypes(exclude=['object']).columns.to_list()
    # Transform categorical columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved at {output_path}")
    
if __name__ == "__main__":
    preprocess(params['input'], params['output'])