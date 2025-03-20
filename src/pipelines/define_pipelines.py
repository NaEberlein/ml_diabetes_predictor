from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

import pipelines.custom_pipeline_components as pipeline_comp

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def create_pipeline_with_imputer_and_classifier(imputer: object, features: list, add_kmeans: bool = True,
                                                classifier: object = RandomForestClassifier(random_state=42)) -> Pipeline:
    """
    Helper function to create pipelines with imputation, KMeans clustering, and a specified classifier.
    
    Parameters:
    imputer: The imputer to use for missing data (e.g., SimpleImputer, KNNImputer, or None).
    features: The features to be passed to the pipeline.
    add_kmeans: Whether to add KMeans clustering features or not (default is True).
    classifier: The classifier to use (default is RandomForestClassifier).
    
    Returns:
    A pipeline with the chosen transformations.
    """
    # Define the scaler and feature preprocessing steps
    scaler = StandardScaler()
    rename_features = pipeline_comp.RenameFeatures()
    preprocess_features = pipeline_comp.PreprocessFeatures(features_no_measurements=["Glucose", "BP", "Skin", "Insulin", "BMI"])
    
    # Create preprocessing pipeline
    preprocessing_pipeline = [
        ("rename", rename_features),
        ("preprocess", preprocess_features)
    ]
    
    # Handle imputer and scaling based on which imputer is used
    if isinstance(imputer, KNNImputer):
        # If KNNImputer, scale before imputation
        steps = preprocessing_pipeline + [("scaler", scaler), ("imputer", imputer)]
    elif isinstance(imputer, pipeline_comp.KNNImputationByGroup):
        # If KNNImputationByGroup, scale before imputation
        steps = preprocessing_pipeline + [("scaler", scaler), ("imputer", imputer)]
        
    elif isinstance(imputer, SimpleImputer):
        # If SimpleImputer, scale after imputation
        steps = preprocessing_pipeline + [("imputer", imputer), ("scaler", scaler)]
    else:
        # If no imputer, just preprocess and scale
        steps = preprocessing_pipeline + [("scaler", scaler)]
    
    # Optionally add KMeans clustering features
    if add_kmeans:
        steps.append(("add_kmeans", pipeline_comp.AddKMeansClusterFeatures(k=2, features=features)))
    
    # Use the provided classifier or default to RandomForest
    steps.append(("classifier", classifier))
    
    return Pipeline(steps)


def pipeline_before_classifier(pipeline : Pipeline) -> Pipeline:
    """ Returns Pipeline with all steps before classifier

    Parameters:
    pipeline: complete pipeline (can include the classifier).

    Return:
    The pipeline with all the steps before the classifier.
    """
    steps_before_classifier = [(name, step) for name, step in pipeline.named_steps.items() if name != "classifier"]
    return Pipeline(steps_before_classifier)

    
def define_pipelines(features: list) -> dict:
    """
    Returns a dictionary of pipelines for various imputation methods
    and classification models.
    
    Parameters:
    features: The list of features to be used in the pipeline.
    
    Returns:
    A dictionary where each key is a pipeline name and each value is the corresponding pipeline object.
    """
    pipelines = {
        "knn_pca": create_pipeline_with_imputer_and_classifier(
            pipeline_comp.KNNImputationByGroup(columns=features, n_neighbors=10, weights="uniform"), features),
        
        "knn": create_pipeline_with_imputer_and_classifier(
            KNNImputer(n_neighbors=10, weights="uniform"), features),
        
        "mean": create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="mean"), features),
        
        "median": create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="median"), features),
        
        "no_imputer_no_kmeans": create_pipeline_with_imputer_and_classifier(
            None, features, add_kmeans=False),
        
        "knn_pca_no_kmeans": create_pipeline_with_imputer_and_classifier(
            pipeline_comp.KNNImputationByGroup(columns=features, n_neighbors=10, weights="uniform"), features, add_kmeans=False),
        
        "knn_no_kmeans": create_pipeline_with_imputer_and_classifier(
            KNNImputer(n_neighbors=10, weights="uniform"), features, add_kmeans=False),
        
        "mean_no_kmeans": create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="mean"), features, add_kmeans=False),
        
        "median_no_kmeans": create_pipeline_with_imputer_and_classifier(
            SimpleImputer(strategy="median"), features, add_kmeans=False),

    }
    
    return pipelines
