import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """ Load dataset and remove non-numeric columns """
    df = pd.read_csv(filepath)

    # Identify and drop non-numeric columns (like dates or text)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        df = df.drop(columns=non_numeric_cols)

    # Convert all remaining data to float
    df = df.astype(np.float64)
    
    return df

def split_data(df, test_size=0.3, random_state=42):
    """ Split dataset into training and testing sets """
    X = df.drop(columns=["silica_concentrate"])  # Features
    y = df["silica_concentrate"]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, output_folder):
    """ Save split datasets """
    os.makedirs(output_folder, exist_ok=True)
    X_train.to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "y_test.csv"), index=False)
    print("Data successfully saved!")

def main():
    input_filepath = "data/raw/raw.csv"
    output_folder = "data/processed"

    df = load_data(input_filepath)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(X_train, X_test, y_train, y_test, output_folder)

if __name__ == "__main__":
    main()
