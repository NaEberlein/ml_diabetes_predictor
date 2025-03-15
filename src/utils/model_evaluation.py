import pickle
import os

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from typing import Dict, List, Tuple, Union


def create_result_dataframe(grid_search: GridSearchCV, scoring_metrics: List[str]) -> pd.DataFrame:
    """
    Generates a dataframe with mean and standard deviation of train and test scores for each scoring metric.
    
    Parameters:
    grid_search (GridSearchCV): The fitted GridSearchCV object.    
    scoring_metrics  (List[str]): Scoring metrics used in GridSearchCV.

    Returns:
    pd.DataFrame: DataFrame containing mean and std for individual scoring metrics.
    """
    if not hasattr(grid_search, "cv_results_"):
        raise ValueError("GridSearchCV object does not contain results. Ensure it has been fitted.")

    cv_results = grid_search.cv_results_    

    # Defining column names for df
    columns = [f"{prefix}_{metric}" for metric in scoring_metrics for prefix in ["mean_train", "mean_test", "std_train", "std_test"]]
    columns += [f"split{split}_train_{metric}" for split in range(grid_search.n_splits_) for metric in scoring_metrics]
    columns += [f"split{split}_test_{metric}" for split in range(grid_search.n_splits_) for metric in scoring_metrics]


    # Create the result dataframe
    result_data = pd.DataFrame(columns=columns)

    # Loop through each metric and fill the dataframe
    for metric in scoring_metrics:
        # Extract the relevant columns for each metric
        mean_train = f"mean_train_{metric}"
        std_train = f"std_train_{metric}"
        mean_test = f"mean_test_{metric}"
        std_test = f"std_test_{metric}"
    
        # Assign values to the result dataframe for each metric individually
        result_data[mean_train] = cv_results[mean_train]
        result_data[std_train] = cv_results[std_train]
        result_data[mean_test] = cv_results[mean_test]
        result_data[std_test] = cv_results[std_test]
    
        # Add per-split accuracy values
        for split in range(grid_search.cv.n_splits):  # Adjusted for cv in grid_search
            # Per split train and test values
            split_train = f"split{split}_train_{metric}"
            split_test = f"split{split}_test_{metric}"
    
            # Add the per-split values to the result dataframe
            result_data[split_train] = cv_results[split_train]
            result_data[split_test] = cv_results[split_test]

        result_data[f"train_test_{metric}_diff"] = abs(result_data[f"mean_train_{metric}"] - result_data[f"mean_test_{metric}"])

    return result_data


def filter_best_models(result_data: pd.DataFrame, grid_search: GridSearchCV, metric: str,
                       score_diff_threshold: float = 0.03, std_test_threshold: float = 0.03
                      ) -> Tuple[pd.Series, Dict[str, Union[str, float]], int]:  
    """
    Filters models based on generalization and stability criteria.
    Returns the best model by test score for specific scoring metric.

    Parameters:
    result_data (pd.DataFrame): DataFrame containing the grid search results.
    grid_search (GridSearchCV): The fitted GridSearchCV object (needed to find the chosen hyperparameters).
    metric (str): Scoring metric on which the best model is selected.
    score_diff_threshold (float): Maximum allowed train-test score difference.
    std_test_threshold (float): Maximum allowed test score standard deviation.
    
    Returns:
    Tuple[pd.Series, Dict[str, Union[str, float]], int]: Best model values, hyperparameters, and index.
    """
    # Apply the filters for good generalisation and stability
    filtered_results = result_data[
        (result_data[f"train_test_{metric}_diff"] <= score_diff_threshold) &
        (result_data[f"std_test_{metric}"] <= std_test_threshold)
    ]

    if filtered_results.empty:
        print("No models met the stability and generalization criteria. Selecting the best overall model.")        
        best_model_idx = result_data[[f"train_test_{metric}_diff", f"std_test_{metric}"]].sum(axis=1).idxmin()
        filtered_results = result_data.copy()
    else:
        best_model_idx = filtered_results[f"mean_test_{metric}"].idxmax()
        
    best_model_values = filtered_results.loc[best_model_idx]
    best_hyperparameters = grid_search.cv_results_["params"][best_model_idx]

    return best_model_values, best_hyperparameters, best_model_idx



def select_best_model(best_models: Dict[str, Tuple[pd.Series, Dict[str, Union[str, float]], int]], 
                      scoring_metrics: List[str]) -> Tuple[str, pd.Series, Dict[str, Union[str, float]], int]:
    """
    Selects the best model based on the lowest sum of train-val differences (good generalisation).
    In case of a tie, selects the model with the lowest standard deviation sum (good stability).

    Parameters:
    best_models (dict): A dictionary where keys are scoring metrics and values are tuples of 
                        (best_model_values, best_hyperparameters, index).
    scoring_metrics (List[str]): A list of scoring metrics used for selection.

    Returns:
    Tuple: (best_metric, best_model_values, best_hyperparameters, index)
    """
    best_metric, best_model = min(
        best_models.items(),
        key=lambda item: (
            sum(item[1][0][f"train_test_{metric}_diff"] for metric in scoring_metrics),  # first select based on diff
            sum(item[1][0][f"std_test_{metric}"] for metric in scoring_metrics)  # second (tie breaker) select based on std
        )
    )
    return best_metric, *best_model




def determine_best_model(pipeline_name : str, grid_results_path: str, scoring_metrics_grid: List[str],
                          score_diff_threshold: float = 0.05, std_test_threshold: float = 0.05) -> tuple:
    """
    Determines the best model based on grid search results.
    Optionally saves the best hyperparameters to a file using pickle.

    Parameters:
    grid_results_path: Path to the saved grid search results.
    scoring_metrics_grid: List of metrics to evaluate the best model.
    score_diff_threshold: Minimum difference required between train and validation score.
    std_test_threshold: Threshold for standard deviation of test score.

    Returns:
    best_hyperparameters and scoring values of the best model
    """
    
    # Load grid search results
    with open(f"{grid_results_path}/grid_search_result_{pipeline_name}.pkl", "rb") as f:
        grid_search_results = pickle.load(f, fix_imports=True)

    # Create DataFrame from grid search results
    df_results = create_result_dataframe(grid_search_results, scoring_metrics_grid)

    # Filter and select the best models for each metric
    best_models = {}
    for metric in scoring_metrics_grid:
        best_model_values, best_params, best_model_idx = filter_best_models(
            df_results, grid_search_results, metric, score_diff_threshold, std_test_threshold)
        best_models[metric] = (best_model_values, best_params, best_model_idx)

    # Select the best model overall (based on the provided metrics)
    best_metric, best_model_values, best_hyperparameters, best_model_idx = select_best_model(
        best_models, scoring_metrics_grid)

    # values of best model 
    best_model_scores = df_results.iloc[best_model_idx]
    return best_hyperparameters, best_model_scores


def save_best_params(best_hyperparameters: dict, pipeline_name: str, save_directory: str = "../src/models/best_params") -> None:
    """
    Saves the best hyperparameters to a pickle file.

    Parameters:
    best_hyperparameters (dict): The best hyperparameters to save.
    pipeline_name (str): Name of the pipeline (used for filename).
    save_directory (str): Directory to save the file. Defaults to '../src/models/best_params'.
    """

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Define the save file path
    save_filename = f"{pipeline_name}_best_params.pkl"
    save_filepath = os.path.join(save_directory, save_filename)

    # Save the hyperparameters using pickle
    with open(save_filepath, "wb") as f:
        pickle.dump(best_hyperparameters, f)

    print(f"Best hyperparameters saved to {save_filepath}")


def evaluate_and_plot_best_model(pipeline: Pipeline, best_hyperparameters: Dict[str, any], X_train: pd.DataFrame,
                                 y_train: pd.Series, features: List[str], scoring_metrics_grid: List[str],
                                 n_splits: int, df_results: pd.DataFrame, best_model_idx: int) -> None:
    """
    Evaluates and plots the best model's cross-validation results and feature importance.

    Parameters:
    pipeline: The model pipeline to be used.
    best_hyperparameters: The best hyperparameters for the model.
    X_train: The training features.
    y_train: The training target.
    features: List of feature names.
    scoring_metrics_grid: List of scoring metrics to use for evaluation.
    n_splits: Number of cross-validation splits.
    df_results: The dataframe with grid search results.
    best_model_idx: The index of the best model from the results.
    """
    # Update the pipeline with the best hyperparameters
    pipeline.set_params(**best_hyperparameters)
    display(pipeline)  # Jupyter's way to show formatted output

    # Plot the cross-validation scores
    plot_train.plot_cv_scores_for_best_model(df_results, best_model_idx, scoring_metrics_grid, n_splits)

    # Plot the feature importance
    plot_train.plot_best_model_feature_importance(best_hyperparameters, pipeline, X_train, y_train, features)

    
def print_best_params(best_params: Dict[str, Union[str, float]]) -> None:
    """
    Prints the best hyperparameters.

    Parameters:
    best_params (Dict[str, Union[str, float]]): Best hyperparameters.
    """
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")



def evaluate_model_performance(best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Calculates and prints evaluation metrics (Accuracy, F1, ROC AUC, Precision, Recall) for both training and testing data.

    Parameters:
    best_pipeline (Pipeline): The trained pipeline with preprocessing and classifier.
    X_train (pd.DataFrame): The feature data for training.
    y_train (pd.Series): The true labels for the training data.
    X_test (pd.DataFrame): The feature data for testing.
    y_test (pd.Series): The true labels for the testing data.
    """
    # Predict on the test and train data
    y_pred = best_pipeline.predict(X_test)
    y_train_pred = best_pipeline.predict(X_train)
    y_pred_proba = best_pipeline.predict_proba(X_test)
    y_train_pred_proba = best_pipeline.predict_proba(X_train)
    # Print the evaluation results
    print(f"Evaluation Metrics (Test Data):\n"
          f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}\n"
          f"F1 Score: {round(f1_score(y_test, y_pred), 4)}\n"
          f"ROC AUC: {round(roc_auc_score(y_test, y_pred_proba[:,1]), 4)}\n"
          f"Precision: {round(precision_score(y_test, y_pred), 4)}\n"
          f"Recall: {round(recall_score(y_test, y_pred), 4)}\n")
    
    print(f"Evaluation Metrics (Train Data):\n"
          f"Accuracy: {round(accuracy_score(y_train, y_train_pred), 4)}\n"
          f"F1 Score: {round(f1_score(y_train, y_train_pred), 4)}\n"
          f"ROC AUC: {round(roc_auc_score(y_train,  y_train_pred_proba[:,1]), 4)}\n"
          f"Precision: {round(precision_score(y_train, y_train_pred), 4)}\n"
          f"Recall: {round(recall_score(y_train, y_train_pred), 4)}")