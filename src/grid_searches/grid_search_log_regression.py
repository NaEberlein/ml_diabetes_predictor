"""
This script performs hyperparameter tuning for different machine learning pipelines with Logistic Regression
using grid search and stratified 10-fold cross-validation.

Steps:
1. Loads preprocessed training data from a CSV file.
2. Imports a set of machine learning pipelines.
3. Performs hyperparameter tuning using GridSearchCV with Stratified K-fold cross-validation.
4. Evaluates models based on scoring metrics: 'accuracy', 'roc_auc', and 'f1'.
5. Saves the results of the grid search to a pickle file.

The hyperparameter grid specifies ranges for parameters such as the regularization strength,
solver type, and maximum iterations, among others. The results of each pipeline's grid search 
are saved to a specified directory for later analysis.
"""

import logging
import pipelines.define_pipelines_log_regression as pipeline_def  
from utils import hyperparams_tuning as tune

logging.basicConfig(filename="grid_search_log_regression.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load training data
    X_train, y_train = tune.load_train_data()

    # Define pipelines
    pipelines_dict = pipeline_def.define_pipelines()

    # Hyperparameter grid and scoring metrics
    scoring_metrics = ['accuracy', 'roc_auc', 'f1', 'recall', 'precision']
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 50],  # Regularization strength
        "classifier__solver": ['liblinear', 'lbfgs'],  # Solvers for Logistic Regression
        "classifier__max_iter": [100, 200, 500],  # Max iterations for convergence
        "classifier__penalty": ['l2'],  # Penalty type for regularization
        "classifier__class_weight": ['balanced', None],  # Class weighting
    }

    # Perform grid search for each pipeline
    for pipeline_name, pipeline in pipelines_dict.items():
        logger.info(f"Processing pipeline: {pipeline_name}")
        grid_search = tune.perform_grid_search(pipeline, param_grid, scoring_metrics, X_train, y_train)
        
        # Save results only if grid search was successful
        tune.save_grid_search_results(pipeline_name, grid_search, save_path="../../results/grid_search_results_log_regression")
