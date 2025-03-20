"""
This script performs hyperparameter tuning for different machine learning pipelines with Random Forests
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


import logging
import pipelines.define_pipelines_random_forest as pipeline_def
from utils import hyperparams_tuning as tune

logging.basicConfig(filename="grid_search_random_forest.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    X_train, y_train = tune.load_train_data()

    pipelines_dict = pipeline_def.define_pipelines()

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
        grid_search = tune.perform_grid_search(pipeline, param_grid, scoring_metrics, X_train, y_train)
        
        # Save results only if grid search was successful
        tune.save_grid_search_results(pipeline_name, grid_search, save_path="../results/grid_search_results_random_forest")
