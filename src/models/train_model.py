import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

def load_data(input_folder):
    """ Load training data """
    X_train = pd.read_csv(os.path.join(input_folder, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_folder, "y_train.csv"))
    return X_train, y_train.values.ravel()

def load_best_params(param_file):
    """ Load best parameters from GridSearch results """
    with open(param_file, "rb") as f:
        best_params = pickle.load(f)
    return best_params

def train_model(X_train, y_train, best_params):
    """ Train model using the best hyperparameters """
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, output_path):
    """ Save trained model """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {output_path}")

def main():
    input_folder = "data/processed"
    param_file = "models/best_params.pkl"
    model_file = "models/trained_model.pkl"

    X_train, y_train = load_data(input_folder)
    best_params = load_best_params(param_file)
    model = train_model(X_train, y_train, best_params)
    save_model(model, model_file)

if __name__ == "__main__":
    main()
