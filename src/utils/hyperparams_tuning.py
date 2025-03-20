import os
import logging
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline  
logger = logging.getLogger(__name__)



target = "Outcome"
features_org = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']


def load_train_data() -> pd.DataFrame:
    """
    Loads the preprocessed training data from a CSV file.
    Validates the existence of the file before loading.
    
    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the data is empty.
    """
    data_file = "../../data/preprocessed_diabetes_train.csv"
    
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


def perform_grid_search(pipeline, param_grid: Dict[str, Any], scoring_metrics: list,
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


def save_grid_search_results(pipeline_name: str, grid_search: GridSearchCV, save_path: str = "../results/hyperparameter_tuning_results") -> None:
    """
    Saves the grid search results to a pickle file.

    Parameters:
    pipeline_name: The name of the pipeline (to be used in the filename).
    grid_search: The grid search object containing the results to be saved.
    save_path: The directory path where the results should be saved (default is `../results/hyperparameter_tuning_results`).
    """
    if grid_search is None:
        logger.error(f"No grid search results to save for {pipeline_name}.")
        return
        
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Generate the full path for the result file
    result_filename = os.path.join(save_path, f"grid_search_result_{pipeline_name}.pkl")    
    try:
        with open(result_filename, "wb") as f:
            pickle.dump(grid_search, f)
        logger.info(f"Grid search results for {pipeline_name} saved to {result_filename}.")
    except Exception as e:
        logger.error(f"Error while saving grid search results for {pipeline_name}: {str(e)}")
