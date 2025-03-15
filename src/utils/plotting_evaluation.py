import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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
    plt.figure(figsize=(7, 5))
    plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Most important features on top
    plt.show()
    plt.close()


def plot_best_model_feature_importance(best_hyperparameters: dict, pipeline: Pipeline,
                                       X_train : pd.DataFrame, y_train : pd.DataFrame, features: list) -> None:
    """
    Plots the feature importances for the best model based on the hyperparameters.
    
    Parameters:
    best_hyperparameters (dict): Best hyperparameters for the model.
    pipeline (Pipeline): The pipeline with preprocessing steps.
    X_train: Training features.
    y_train: Training labels.
    features (list): List of feature names.
    """
    
    # Set the best hyperparameters for the pipeline
    best_pipeline = pipeline.set_params(**best_hyperparameters)
    
    # Train the model with the best hyperparameters
    best_pipeline.fit(X_train, y_train)
    
    # Get the classifier model from the pipeline
    classifier = best_pipeline.named_steps.get("classifier", None)
    
    if classifier is None:
        print("No classifier found in the pipeline!")
        return
    
    # Check if the classifier has feature_importances_ attribute
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
    else:
        print("The model does not support feature importances.")
        return
    
    # If KMeans clustering is part of the pipeline, add "Cluster" to the feature names
    if "add_kmeans" in best_pipeline.named_steps:
        feature_names = features + ["Cluster"]  # Add the new "Cluster" feature
    else:
        feature_names = features
    
    plot_feature_importance(feature_importances, feature_names)



def plot_shap_values_for_class_1(best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Plots the SHAP summary plot for class 1 (diabetic) using the given pipeline and training data.

    Parameters:
    best_pipeline (Pipeline): The trained pipeline with preprocessing and classification steps.
    X_train (pd.DataFrame): The training feature data.
    y_train (pd.Series): The target labels for the training data.
    """
    
    # Extract preprocessing pipeline and fit it
    preprocessing_pipeline = best_pipeline[:-1]
    preprocessing_pipeline.fit(X_train, y_train)
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)  # Apply transformation

    # Get the trained model
    trained_model = best_pipeline[-1]  # The classifier part of the pipeline
    
    # Create SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X_train_transformed)
    
    # Plot SHAP summary for class 1 (diabetic)
    shap.summary_plot(shap_values[:, :, 1], X_train_transformed,  plot_size=(7, 5))  # For class 1


def plot_confusion_matrix(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plots confusion matrices for both test and train data side by side.

    Parameters:
    pipeline (Pipeline): The trained pipeline with preprocessing and classifier.
    X_train (np.ndarray): Features for the training data.
    y_train (np.ndarray or pd.Series): True labels for the training data.
    X_test (np.ndarray): Features for the test data.
    y_test (np.ndarray or pd.Series): True labels for the test data.
    """
    # Predict on the test and train data
    y_test_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)
    
    # Generate confusion matrices for both test and train data
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_train = confusion_matrix(y_train, y_train_pred)
    
    # Set up the subplot for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Plot test confusion matrix
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names.values(), yticklabels=class_names.values(), ax=axes[0])
    axes[0].set_title('Confusion Matrix - Test Data')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    
    # Plot train confusion matrix
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", xticklabels=class_names.values(), yticklabels=class_names.values(), ax=axes[1])
    axes[1].set_title('Confusion Matrix - Train Data')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')
    
    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_roc_curve(best_pipeline, X_train, y_train, X_test, y_test):
    """
    Plots the ROC curve for the test data and computes the AUC score.

    Parameters:
    best_pipeline (Pipeline): The trained pipeline with preprocessing and classifier.
    X_train (np.ndarray or pd.DataFrame): Features for the training data.
    y_train (np.ndarray or pd.Series): True labels for the training data.
    X_test (np.ndarray or pd.DataFrame): Features for the test data.
    y_test (np.ndarray or pd.Series): True labels for the test data.
    """
     # Predict probabilities for test and train data
    y_test_pred_proba = best_pipeline.predict_proba(X_test)
    y_train_pred_proba = best_pipeline.predict_proba(X_train)

    # Compute ROC curve and AUC score for test data
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred_proba[:, 1])
    roc_auc_test = auc(fpr_test, tpr_test)

    # Compute ROC curve and AUC score for train data
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred_proba[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)

    plt.figure(figsize=(4,4))
    plt.plot(fpr_test, tpr_test, lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
    plt.plot(fpr_train, tpr_train, lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def plot_precision_recall_curve(best_pipeline, X_train, y_train, X_test, y_test):
    """
    Plots the Precision-Recall curve for both test and train data.

    Parameters:
    best_pipeline (Pipeline): The trained pipeline with preprocessing and classifier.
    X_train (np.ndarray or pd.DataFrame): Features for the training data.
    y_train (np.ndarray or pd.Series): True labels for the training data.
    X_test (np.ndarray or pd.DataFrame): Features for the test data.
    y_test (np.ndarray or pd.Series): True labels for the test data.
    """
    # Predict probabilities for test and train data
    y_test_pred_proba = best_pipeline.predict_proba(X_test)
    y_train_pred_proba = best_pipeline.predict_proba(X_train)

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba[:, 1])
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba[:, 1])

    plt.figure(figsize=(4,4))
    plt.plot(recall_test, precision_test, lw=2, label=f'Test PR curve')
    plt.plot(recall_train, precision_train, lw=2, label=f'Train PR curve')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    plt.close()