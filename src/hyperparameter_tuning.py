"""
This script performs hyperparameter tuning for different machine learning pipelines 
using grid search and stratiefied 10-fold cross validation.

Steps:
1. Loads preprocessed training data from a CSV file.
2. Imports a set of machine learning pipelines.
3. Performs hyperparameter tuning using GridSearchCV with Stratified K-fold cross-validation.
4. Evaluates models based on scoring metrics: 'accuracy', 'roc_auc', and 'f1'.
5. Saves the results of the grid search to a pickle file.

The hyperparameter grid specifies ranges for parameters such as the number of estimators, 
max depth, and class weight, among others. The results of each pipeline's grid search 
are saved to a specified directory for later analysis.

"""


import os
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

import pipelines.define_pipelines as pipeline_def


logging.basicConfig(filename="hyperparameter_tuning.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

target = "Outcome"
features = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]
features_org = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
def load_train_data() -> pd.DataFrame:
    """
    Loads the preprocessed training data from a CSV file.
    Validates the existence of the file before loading.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the data is empty.
    """
    data_file = "../data/preprocessed_diabetes_train.csv"
    
    if not os.path.exists(data_file):
        logger.error(f"The training data file at {data_file} was not found.")
        raise FileNotFoundError(f"The training data file at {data_file} was not found.")

    data = pd.read_csv(data_file)
    
    if data.empty:
        logger.error(f"The training data at {data_file} is empty.")
        raise ValueError(f"The training data at {data_file} is empty.")

    X = data[features_org]
    y = data[target]
    return X, y


def perform_grid_search(pipeline: Pipeline, param_grid: Dict[str, Any], scoring_metrics: list,
                        X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Performs hyperparameter tuning using grid search.
    """
    logger.info(f"Starting grid search with scoring metrics: {scoring_metrics}")
    
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=stratified_kfold,
                               scoring=scoring_metrics, return_train_score=True, n_jobs=10, verbose=1, refit=False)
    grid_search.fit(X_train, y_train)

    return grid_search


def save_grid_search_results(pipeline_name: str, grid_search: GridSearchCV) -> None:
    """
    Saves the grid search results to a pickle file.
    """
    if grid_search is None:
        logger.error(f"No grid search results to save for {pipeline_name}.")
        return
        
    results_dir = os.path.join("../results", "hyperparameter_tuning_results")
    os.makedirs(results_dir, exist_ok=True)
    
    result_filename = os.path.join(results_dir, f"grid_search_result_{pipeline_name}.pkl")    
    try:
        with open(result_filename, "wb") as f:
            pickle.dump(grid_search, f)
        logger.info(f"Grid search results for {pipeline_name} saved to {result_filename}.")
    except Exception as e:
        logger.error(f"Error while saving grid search results for {pipeline_name}: {str(e)}")




if __name__ == "__main__":
    X_train, y_train = load_train_data()

    pipelines_dict = pipeline_def.define_pipelines(features)

    # Hyperparameter grid and scoring metrics
    scoring_metrics = ['accuracy', 'roc_auc', 'f1']
    param_grid = {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_split": [5, 10, 20, 30],
        "classifier__min_samples_leaf": [2, 5, 10],
        "classifier__max_features": ["log2", "sqrt", 0.5],
        "classifier__class_weight": ["balanced"],
        "classifier__bootstrap": [True],
        "classifier__max_samples": [0.6, 0.7, 0.8],
        "classifier__criterion": ["gini", "entropy"],
    }

    # Perform grid search for each pipeline
    for pipeline_name, pipeline in pipelines_dict.items():
        logger.info(f"Processing pipeline: {pipeline_name}")
        grid_search = perform_grid_search(pipeline, param_grid, scoring_metrics, X_train, y_train)
        
        # Save results only if grid search was successful
        save_grid_search_results(pipeline_name, grid_search)