from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

import pipelines.custom_pipeline_components as pipeline_comp


features = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]

def create_pipeline_with_imputer_and_classifier(imputer: object, add_kmeans: bool = False,
                                                classifier: object = RandomForestClassifier(random_state=42)) -> Pipeline:
    """
    Helper function to create pipelines with imputation, KMeans clustering, and a specified classifier.
    
    Parameters:
    imputer: The imputer to use for missing data (e.g., SimpleImputer, KNNImputer, or None).
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
    
    # Use the provided classifier (default RandomForest)
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
