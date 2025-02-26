import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def normalize_data(input_folder, output_folder):
    """ Normalize feature data using StandardScaler """
    X_train = pd.read_csv(os.path.join(input_folder, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_folder, "X_test.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    os.makedirs(output_folder, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(output_folder, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_folder, "X_test_scaled.csv"), index=False)
    print("Normalization complete and saved.")

def main():
    input_folder = "data/processed"
    output_folder = "data/processed"

    normalize_data(input_folder, output_folder)

if __name__ == "__main__":
    main()
