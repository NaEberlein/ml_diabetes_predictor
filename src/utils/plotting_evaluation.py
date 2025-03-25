import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression

class_names = {0: "Not Diabetic", 1: "Diabetic"}


def plot_cv_scores_subplots(val_scores: list, train_scores: list, metric: str, ax) -> None:
    """
    Plots the cross-validation results for train and validation scores on a given subplot axis.

    Parameters:
    val_scores (List[float]): Validation scores from cross-validation.
    train_scores (List[float]): Training scores from cross-validation.
    metric (str): The name of the scoring metric used.
    ax (matplotlib.axes.Axes): The subplot axis to plot on.
    """
    ax.plot(val_scores, label="Validation", marker="o", color="green", linestyle="-", linewidth=2, markersize=4)
    ax.plot(train_scores, label="Train", marker="o", color="blue", linestyle="-", linewidth=2, markersize=4)

    # Plot average train and validation scores with standard deviation reference lines
    ax.axhline(
        y=np.mean(train_scores), color="blue", linestyle="--",
    )
    ax.axhline(
        y=np.mean(val_scores), color="green", linestyle="--",
    )
    # calc duff between train and val scores 
    diff = np.mean(train_scores) - np.mean(val_scores)
    ax.legend(loc="upper right", labels=[
    f"Train (Avg: {np.mean(train_scores):.2f} ± {np.std(train_scores):.2f})",
    f"Validation (Avg: {np.mean(val_scores):.2f} ± {np.std(val_scores):.2f})",
    ])

    ax.set_xlabel("Fold Number")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim((min(val_scores) - 0.1, max(train_scores) + 0.2))
    # ax.set_title(f"Cross-Validation: {metric}")

def plot_cv_scores_for_best_model(best_model_scores: pd.DataFrame, scoring_metrics_grid: list, 
                                  n_splits: int) -> None:
    """
    Plots the cross-validation scores for the best model in one row using subplots.

    Parameters:
    best_model_scores (pd.DataFrame): DataFrame containing the results of the best model.
    scoring_metrics_grid (List[str]): List of scoring metrics to plot (e.g., accuracy, precision).
    n_splits (int): Number of cross-validation splits.
    """
    num_metrics = len(scoring_metrics_grid)
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 6, 5))
    fig.suptitle("Cross-Validation Scores for Best Model", fontsize=16)  # Title for all subplots
    # Ensure axes is iterable even for a single metric
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, scoring_metrics_grid):
        # Extract validation scores for each split for the given metric
        val_scores = [best_model_scores[f"split{n}_test_{metric}"] for n in range(n_splits)]
        train_scores = [best_model_scores[f"split{n}_train_{metric}"] for n in range(n_splits)]
        
        # Plot on respective subplot
        plot_cv_scores_subplots(val_scores, train_scores, metric, ax)

    plt.tight_layout()
    plt.show()
    plt.close()



def plot_feature_importance(importances : np.ndarray, feature_names : list) -> None:
    """
    Plots feature importance for a classifier inside a pipeline, considering 
    selected features and transformations applied before the classifier.

    Parameters:
    importances (array): The importance values for each feature.
    feature_names (list): The list of feature names.
    """

    if len(importances) != len(feature_names):
        raise ValueError("The number of importances must match the number of feature names.")
    
    sorted_indices = np.argsort(importances)[::-1]  # Sort in descending order
    sorted_importances = importances[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]
    
    # Plot feature importances
    plt.figure(figsize=(6, 5))
    plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Most important features on top
    plt.show()
    plt.close()


def plot_best_model_feature_importance(best_pipeline: Pipeline,
                                       X_train : pd.DataFrame, y_train : pd.DataFrame, features: list) -> None:
    """
    Plots the feature importances for the best model based on the hyperparameters.
    
    Parameters:
    best_pipeline (Pipeline): The fitted pipeline with preprocessing steps.
    X_train: Training features.
    y_train: Training labels.
    features (list): List of feature names.
    """
    
    # Train the model with the best hyperparameters
    best_pipeline.fit(X_train, y_train)
    
    # Get the classifier model from the pipeline
    classifier = best_pipeline.named_steps.get("classifier", None)
    
    if classifier is None:
        print("No classifier found in the pipeline!")
        return
    
    # Check if the classifier has feature importances (tree-based models)
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
    
    # Check if the classifier has coefficients (linear models like Logistic Regression)
    elif hasattr(classifier, 'coef_'):
        feature_importances = np.abs(classifier.coef_).flatten()  # Take absolute values for importance
    else:
        print("The model does not support feature importances or coefficients.")
        return
    
    # If KMeans clustering is part of the pipeline, add "Cluster" to the feature names
    if "add_kmeans" in best_pipeline.named_steps:
        feature_names = features + ["Cluster"]  # Add the new "Cluster" feature
    else:
        feature_names = features
    
    plot_feature_importance(feature_importances, feature_names)


def plot_shap_values_for_class_1(best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, features: list) -> None:
    """
    Plots the SHAP summary plot for class 1 (diabetic) using the given pipeline and training data.

    Parameters:
    best_pipeline (Pipeline): The trained pipeline with preprocessing and classification steps.
    X_train (pd.DataFrame): The training feature data.
    y_train (pd.Series): The target labels for the training data.
    features (list): list of all the features.
    """
    
    # Extract preprocessing pipeline and fit it
    preprocessing_pipeline = best_pipeline[:-1]
    preprocessing_pipeline.fit(X_train, y_train)
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)  # Apply transformation

    # Get the trained model
    trained_model = best_pipeline[-1]  # The classifier part of the pipeline

    # If KMeans clustering is part of the pipeline, add "Cluster" to the feature names
    if "add_kmeans" in best_pipeline.named_steps:
        feature_names = features + ["Cluster"]  # Add the new "Cluster" feature
    else:
        feature_names = features

    
    # Create SHAP explainer and calculate SHAP values
    if isinstance(trained_model, LogisticRegression):
        explainer = shap.Explainer(trained_model, X_train_transformed) 
        shap_values = explainer(X_train_transformed)
        shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, plot_size=(6, 5)) # For class 1
    else:    
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(X_train_transformed)
        # Plot SHAP summary for class 1 (diabetic)
        shap.summary_plot(shap_values[:, :, 1], X_train_transformed, feature_names=feature_names, plot_size=(6, 5))  # For class 1


def plot_learning_curve(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame,
                        scoring_metric: str = "accuracy") -> None:
    """ 
    Plots the learning curve of the train and validation data set for varying size of data.
    Train data varies between 10% to 100% of the full training data (without the validation data).

    Parameters:
    pipeline: Pipeline including the classifier model.
    X_train, y_train: Training data.
    scoring_metric: Metric for which the learning curve is plotted (default accuracy).
    """
    
    # Determine Learning curve for various numbers of training data sizes
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # same as in hyperparameter tuning to prevent data leakage
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=stratified_kfold, scoring=scoring_metric, train_sizes=np.linspace(0.1, 1., 20)
    )
    # Calculate mean and standard deviation for both training and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    
    # Plot the learning curve
    plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
    plt.plot(train_sizes, val_mean, label="Validation Score", color="green", marker="s")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="green", alpha=0.1)
    
    # Adding labels and title
    plt.xlabel("Training Size")
    plt.ylabel(scoring_metric.replace('_',' '))
    plt.title(f"Learning Curve ({scoring_metric.replace('_',' ')})")
    plt.legend()
    plt.show()
