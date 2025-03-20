from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer

import pipelines.custom_pipeline_components as pipeline_comp
import utils.pipeline as pipe_utils

features = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]

def define_pipelines() -> dict:
    """
    Returns a dictionary of pipelines for various imputation methods
    and a Random Forest model.
    
    Returns:
    A dictionary where each key is a pipeline name and each value is the corresponding pipeline object.
    """
    pipelines = {
        "knn_pca": pipe_utils.create_pipeline_with_imputer_and_classifier(
            pipeline_comp.KNNImputationByGroup(columns=features, n_neighbors=10, weights="uniform"),
            add_kmeans=True,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "knn": pipe_utils.create_pipeline_with_imputer_and_classifier(
            KNNImputer(n_neighbors=10, weights="uniform"),
            add_kmeans=True,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "mean": pipe_utils.create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="mean"),
            add_kmeans=True,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "median": pipe_utils.create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="median"),
            add_kmeans=True,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "no_imputer_no_kmeans": pipe_utils.create_pipeline_with_imputer_and_classifier(
            None, 
            add_kmeans=False,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "knn_pca_no_kmeans": pipe_utils.create_pipeline_with_imputer_and_classifier(
            pipeline_comp.KNNImputationByGroup(columns=features, n_neighbors=10, weights="uniform"),
            add_kmeans=False,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "knn_no_kmeans": pipe_utils.create_pipeline_with_imputer_and_classifier(
            KNNImputer(n_neighbors=10, weights="uniform"),
            add_kmeans=False,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "mean_no_kmeans": pipe_utils.create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="mean"),
            add_kmeans=False,
            classifier=RandomForestClassifier(random_state=42)
        ),
        
        "median_no_kmeans": pipe_utils.create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="median"),
            add_kmeans=False,
            classifier=RandomForestClassifier(random_state=42)
        ),
    }
    
    return pipelines
