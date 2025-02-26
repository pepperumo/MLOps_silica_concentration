import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

def load_data(input_folder):
    """ Load training data """
    X_train = pd.read_csv(os.path.join(input_folder, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_folder, "y_train.csv"))
    return X_train, y_train.values.ravel()  # Flatten y_train

def grid_search(X_train, y_train):
    """ Perform Grid Search to find best hyperparameters """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_

def save_best_params(params, output_path):
    """ Save best parameters as a .pkl file """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Best parameters saved to {output_path}")

def main():
    input_folder = "data/processed"
    output_file = "models/best_params.pkl"

    X_train, y_train = load_data(input_folder)
    best_params = grid_search(X_train, y_train)
    save_best_params(best_params, output_file)

if __name__ == "__main__":
    main()
