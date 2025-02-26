# MLOps Silica Concentration Project

This repository demonstrates a machine learning pipeline using **DVC** and **Git** to predict **silica concentration** from operational parameters in mineral processing.

## Table of Contents
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Setup & Installation](#setup--installation)  
4. [Usage](#usage)  
5. [Pipeline Stages](#pipeline-stages)  
6. [DVC Commands](#dvc-commands)  
7. [Instructions](#instructions)
8. [Data](#data)
9. [Contributing](#contributing)  
10. [License](#license)

---

## Overview
- **Goal**: Model the relationship between flotation parameters and final silica concentration.
- **Data**: Contains features like flow rates, pH, density, etc.
- **Pipeline**:
    1. Split data (train/test).
    2. Normalize features.
    3. Perform hyperparameter tuning (GridSearch).
    4. Train a regression model.
    5. Evaluate model performance (MSE, R²).

---

## Project Structure

```
MLOps_silica_concentration/
├── data/
│   ├── raw/               # Contains raw.csv
│   └── processed/         # Contains train/test data & scaled data
├── metrics/               # Stores evaluation metrics
│   └── scores.json        # Model performance metrics (MSE, R2)
├── models/
│   ├── best_params.pkl    # Hyperparameters from grid search
│   └── gbr_model.pkl      # Trained model artifact
├── src/
│   ├── data/
│   │   ├── data_split.py   # Splits raw data
│   │   └── normalize.py    # Normalizes features
│   └── models/
│       ├── grid_search.py  # Finds best hyperparams
│       ├── training.py     # Trains the model
│       └── evaluate.py     # Evaluates model performance
├── dvc.yaml               # Defines DVC pipeline stages
├── .gitignore
├── .dvcignore
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Setup & Installation

1. **Clone the Repo**  
     ```bash
     git clone https://github.com/<YOUR_USERNAME>/MLOps_silica_concentration.git
     cd MLOps_silica_concentration
     ```

2. **Create a Virtual Environment (Optional but Recommended)**  
     ```bash
     python -m venv env
     source env/bin/activate  # On Linux/Mac
     env\Scripts\activate     # On Windows
     ```

3. **Install Dependencies**  
     ```bash
     pip install -r requirements.txt
     ```

4. **Initialize DVC** (If not already)  
     ```bash
     dvc init
     ```

---

## Usage

### **Run the Pipeline**
```bash
dvc repro
```
This executes all stages (split → normalize → gridsearch → training → evaluate) as defined in `dvc.yaml`.

### **Check the Pipeline Graph**
```bash
dvc dag
```
Displays a DAG of the pipeline in the terminal.

### **View Final Metrics**
```bash
cat metrics/scores.json
```
Shows the MSE and R² scores.

---

## Pipeline Stages

1. **Split**  
     - **Script**: `src/data/data_split.py`  
     - **Inputs**: `data/raw/raw.csv`  
     - **Outputs**: `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

2. **Normalize**  
     - **Script**: `src/data/normalize.py`  
     - **Inputs**: `data/processed/X_train.csv`, `X_test.csv`  
     - **Outputs**: `data/processed/X_train_scaled.csv`, `X_test_scaled.csv`

3. **Grid Search**  
     - **Script**: `src/models/grid_search.py`  
     - **Inputs**: `X_train_scaled.csv`, `y_train.csv`  
     - **Outputs**: `models/best_params.pkl` (hyperparameters)

4. **Training**  
     - **Script**: `src/models/training.py`  
     - **Inputs**: `X_train_scaled.csv`, `y_train.csv`, `best_params.pkl`  
     - **Outputs**: `models/gbr_model.pkl` (trained model)

5. **Evaluate**  
     - **Script**: `src/models/evaluate.py`  
     - **Inputs**: `X_test_scaled.csv`, `y_test.csv`, `gbr_model.pkl`  
     - **Outputs**: `data/prediction.csv`, `metrics/scores.json` (MSE, R²)

---

## DVC Commands

- **Add & Track New Files**  
    ```bash
    dvc add data/raw/new_data.csv
    git add data/raw/new_data.csv.dvc .gitignore
    git commit -m "Add new data"
    ```
- **Push Data/Models to Remote** (e.g., DagsHub)  
    ```bash
    dvc remote add origin https://dagshub.com/<YOUR_USERNAME>/<REPO_NAME>.dvc
    dvc push
    git push origin main
    ```
- **Pull Data/Models**  
    ```bash
    dvc pull
    ```

---

## Instructions

1. **Fork** this repository.
2. **Clone** the forked repository to your local machine.

The submission for this exam will be the link to your repository on DagsHub. Make sure to add `https://dagshub.com/licence.pedago` as a collaborator with read-only rights so that it can be graded.

## Data

Download the dataset from: [https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv](https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv).

---

## Contributing

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add cool feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Create a **Pull Request**.

---

## License

This project is licensed under the **MIT License**.  
Feel free to use and modify the code for your own purposes.

**Enjoy this MLOps workflow with DVC!**
