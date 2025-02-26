import pandas as pd
import pickle
import os
import json
from sklearn.metrics import mean_squared_error, r2_score

def load_data(input_folder):
    """ Load test data """
    X_test = pd.read_csv(os.path.join(input_folder, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(input_folder, "y_test.csv"))
    return X_test, y_test.values.ravel()

def load_model(model_path):
    """ Load trained model """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model and return metrics """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, {"MSE": mse, "R2": r2}

def save_results(y_pred, scores, pred_output, scores_output):
    """ Save predictions and scores """
    os.makedirs(os.path.dirname(pred_output), exist_ok=True)
    os.makedirs(os.path.dirname(scores_output), exist_ok=True)

    # Save predictions
    pd.DataFrame(y_pred, columns=["Predicted"]).to_csv(pred_output, index=False)

    # Save metrics
    with open(scores_output, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"Predictions saved to {pred_output}")
    print(f"Scores saved to {scores_output}")

def main():
    input_folder = "data/processed"
    model_file = "models/trained_model.pkl"
    pred_output = "data/predictions.csv"
    scores_output = "metrics/scores.json"

    X_test, y_test = load_data(input_folder)
    model = load_model(model_file)
    y_pred, scores = evaluate_model(model, X_test, y_test)
    save_results(y_pred, scores, pred_output, scores_output)

if __name__ == "__main__":
    main()
