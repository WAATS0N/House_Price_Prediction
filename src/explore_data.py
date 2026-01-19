import pandas as pd
import numpy as np
import os

def explore_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\n--- Basic Info ---")
    print(df.info())
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Value Counts for Categorical Columns ---")
    categorical_cols = ['area_type', 'availability', 'location', 'size', 'society']
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts().head(10))

if __name__ == "__main__":
    file_path = "data\House_Data.csv"
    if os.path.exists(file_path):
        explore_data(file_path)
    else:
        print(f"Error: {file_path} not found.")
