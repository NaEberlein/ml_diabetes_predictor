import pickle
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score, make_scorer)
from sklearn.model_selection import TunedThresholdClassifierCV, cross_validate


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
    n_splits = grid_search.cv.get_n_splits()  # Get number of splits from cross-validation

    # Prepare dictionary to hold the data for the DataFrame
    result_data = {}

    # Loop through each metric and fill the dictionary
    for metric in scoring_metrics:
        # Extract relevant columns for each metric
        mean_train_col = f"mean_train_{metric}"
        std_train_col = f"std_train_{metric}"
        mean_test_col = f"mean_test_{metric}"
        std_test_col = f"std_test_{metric}"

        # Fill in the dictionary with mean and std columns for train and test
        result_data[mean_train_col] = cv_results[mean_train_col]
        result_data[std_train_col] = cv_results[std_train_col]
        result_data[mean_test_col] = cv_results[mean_test_col]
        result_data[std_test_col] = cv_results[std_test_col]

        # Add per-split accuracy values to the dictionary
        for split in range(n_splits):
            split_train_col = f"split{split}_train_{metric}"
            split_test_col = f"split{split}_test_{metric}"

            result_data[split_train_col] = cv_results[split_train_col]
            result_data[split_test_col] = cv_results[split_test_col]

        # Calculate and add the difference between mean train and test scores
        result_data[f"train_test_{metric}_diff"] = abs(cv_results[mean_train_col] - cv_results[mean_test_col])

    # Convert the result_data dictionary to a DataFrame
    result_df = pd.DataFrame(result_data)

    return result_df

def filter_best_models(result_data: pd.DataFrame, grid_search: GridSearchCV, metric: str,
                       score_diff_threshold: float = 0.05, std_test_threshold: float = 0.05
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
        print(f"[Scoring metric: {metric}] No models met the train - val diff < {score_diff_threshold} "
              f"and std val < {std_test_threshold} criteria. Selecting no model for this metric.")
        return None, None, None

    else:
        best_model_idx = filtered_results[f"mean_test_{metric}"].idxmax()
        
        best_model_values = filtered_results.loc[best_model_idx]
        best_hyperparameters = grid_search.cv_results_["params"][best_model_idx]
    
        return best_model_values, best_hyperparameters, best_model_idx



def select_best_model(best_models: Dict[str, Tuple[pd.Series, Dict[str, Union[str, float]], int]], 
                      selection_metrics: List[str]) -> Tuple[str, pd.Series, Dict[str, Union[str, float]], int]:
    """
    Selects the best model based on the follwing criteria:
    1. mean of selection metrics
    2. lowest mean of train-val differences of selection metrics (good generalisation).
    3. lowest mean standard deviation sum of selection metrics(good stability).

    Parameters:
    best_models (dict): A dictionary where keys are scoring metrics and values are tuples of 
                        (best_model_values, best_hyperparameters, index).
    selection_metrics (List[str]): A list of metrics used for selection.

    Returns:
    Tuple: (best_metric, best_model_values, best_hyperparameters, index)
    """
    model_scores = []

    for metric, (best_model_values, best_hyperparameters, index) in best_models.items():
        if best_model_values is None:  # no valid model found -> skip 
            continue
        
        # mean of test metric, diff train - test metric and std test metric
        mean_test_score = np.mean([best_model_values[f"mean_test_{m}"] for m in selection_metrics])
        mean_train_test_diff = np.mean([best_model_values[f"train_test_{m}_diff"] for m in selection_metrics])
        mean_std_test = np.mean([best_model_values[f"std_test_{m}"] for m in selection_metrics])
        model_scores.append(
            (mean_test_score, mean_train_test_diff, mean_std_test, metric, best_model_values, best_hyperparameters, index)
        )

    
    # find model with highest test score metric, followed by smallest diff between train and test and last based on std on test metric
    model_scores.sort(key=lambda x: (-x[0], x[1], x[2])) 
    
    best_model = model_scores[0][-4:]  # only need metric, best_model_values, best_hyperparameters, index
    return best_model


def load_grid_search_results(grid_results_path: str, pipeline_name: str) -> dict:
    """
    Loads the grid search results from a pickle file.
    
    Parameters:
    grid_results_path (str): Path to the directory containing the grid search result file.
    pipeline_name (str): Name of the pipeline to load the results for.
    
    Returns:
    dict: The loaded grid search results.
    """
    try:
        with open(f"{grid_results_path}/grid_search_result_{pipeline_name}.pkl", "rb") as f:
            return pickle.load(f, fix_imports=True)
    except FileNotFoundError:
        print(f"[Error] Grid search result file not found at {grid_results_path}/grid_search_result_{pipeline_name}.pkl")
        return None
    except Exception as e:
        print(f"[Error] Failed to load grid search results: {str(e)}")
        return None


        

def print_model_metrics(best_models: Dict[str, List[Tuple[pd.DataFrame]]]) -> None:
    """
    Prints the performance metrics of the models chosen based on various metrics.
    
    Parameters:
    - best_models (Dict[str, List[Tuple[pd.DataFrame]]]): A dictionary where keys are the model names,
      and values are lists containing tuples with DataFrames holding the metrics.

    """
    metrics = list(best_models.keys())
    rows = []
    for m in metrics:
        rows.append(f"{m.replace('_',' ').capitalize()} val.")
        rows.append(f"{m.replace('_',' ').capitalize()} diff. train - val.")
    data = pd.DataFrame(index = rows)
    
    # Loop through each model (for each model, extract the metrics)
    for model_name in best_models:
        df_model = best_models[model_name][0]  # Get the DataFrame for the current model
        column_data = []
        for m in metrics:
            column_data.append(f"{df_model[f'mean_test_{m}']:.3f} ± {df_model[f'std_test_{m}']:.3f}")  # mean ± std
            column_data.append(f"{df_model[f'train_test_{m}_diff']:.3f}")  # train-test diff

        # Append the row data to the table data
        data[f"{model_name.replace('_', ' ').capitalize()}"] = column_data    
    multi_index = pd.MultiIndex.from_product([["BEST MODELS BASED ON METRIC"], data.columns])
    
    # Assign MultiIndex to DataFrame
    data.columns = multi_index

   
    print(f"\nComparison of Metrics for Each Model:\n"
          f"{data.to_string()}"
          f"\n\nThe final model is selected based on the highest average validation score across all metrics.")
        


def determine_best_model(pipeline_name : str, grid_results_path: str, selection_metrics: List[str],
                         score_diff_threshold: float = 0.05, std_test_threshold: float = 0.05, printing: bool = False) -> tuple:
    """
    Determines the best model based on grid search results.
    Optionally saves the best hyperparameters to a file using pickle.

    Parameters:
    grid_results_path: Path to the saved grid search results.
    selection_metrics: List of metrics to evaluate the best model.
    score_diff_threshold: Minimum difference required between train and validation score.
    std_test_threshold: Threshold for standard deviation of test score.

    Returns:
    best_hyperparameters and scoring values of the best model
    """
    
    # Load grid search results
    grid_search_results = load_grid_search_results(grid_results_path, pipeline_name)
    
    if grid_search_results is None:
        return None, None

    # Create DataFrame from grid search results
    df_results = create_result_dataframe(grid_search_results, selection_metrics)

    # Filter and select the best models for each metric
    best_models = {}
    for metric in selection_metrics:
        best_model_values, best_params, best_model_idx = filter_best_models(
            df_results, grid_search_results, metric, score_diff_threshold, std_test_threshold)
        best_models[metric] = (best_model_values, best_params, best_model_idx)
    
    # print the results of the best models
    if printing: 
        print(f"Model Selection Criteria applied:\n"
              f"- Highest validation metric score\n"
              f"- Train-validation difference < {score_diff_threshold}\n"
              f"- Validation standard deviation < {std_test_threshold}\n"
              f"\nFor each metric, the best model is selected based on the above criteria.")
        print_model_metrics(best_models)

    # Select the best model overall (based on the provided metrics)
    best_metric, best_model_values, best_hyperparameters, best_model_idx = select_best_model(
        best_models, selection_metrics)


    
    # values of best model 
    best_model_scores = df_results.iloc[best_model_idx]
    return best_hyperparameters, best_model_scores


def save_best_params(best_hyperparameters: dict, pipeline_name: str, classifier_type: str ,
                     save_directory: str = "../src/models/best_params") -> None:
    """
    Saves the best hyperparameters to a pickle file.

    Parameters:
    best_hyperparameters (dict): The best hyperparameters to save.
    pipeline_name (str): Name of the pipeline (used for filename).
    classifier_type (str): Additional flag to add the classifier type.
    save_directory (str): Directory to save the file. Defaults to '../src/models/best_params'.
    """

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Define the save file path
    save_filename = f"{pipeline_name}_best_params_{classifier_type}.pkl"
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



def evaluate_model_performance(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame,
                               X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Calculates and prints evaluation metrics (Accuracy, F1, ROC AUC, Precision, Recall) for both training and testing data.

    Parameters:
    pipeline (Pipeline): The trained pipeline with preprocessing and classifier.
    X_train (pd.DataFrame): The feature data for training.
    y_train (pd.DataFrame): The true labels for the training data.
    X_test (pd.DataFrame): The feature data for testing.
    y_test (pd.DataFrame): The true labels for the testing data.
    """
    # Predict on the test and train data
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)
    y_pred_proba = pipeline.predict_proba(X_test)
    y_train_pred_proba = pipeline.predict_proba(X_train)

    scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # same as in hyperparameter tuning
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
   
    print(f"Evaluation Metrics (Cross-validation Data):")

    for metric in scoring:
        mean = cv_results[f'test_{metric}'].mean()
        std = cv_results[f'test_{metric}'].std()
        print(f"{metric.replace('_',' ').capitalize()}: {mean:.4f} ± {std:.4f}")

    # Print the evaluation results
    print(f"\nEvaluation Metrics (Test Data):\n"
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