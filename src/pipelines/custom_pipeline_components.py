import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class RenameFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer for renaming feature columns based on a fixed dictionary.
    """
    def __init__(self):
        # Hard-code the feature mapping inside the class
        self.feature_mapping = {
            "SkinThickness": "Skin",
            "BloodPressure": "BP",
            "DiabetesPedigreeFunction": "DPF"
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the features in the DataFrame based on the feature_mapping.
        
        Parameters:
        - X: Input data (pandas DataFrame).
        
        Returns:
        - X: Transformed data with renamed feature columns.
        """
        X_renamed = X.copy()
        X_renamed = X_renamed.rename(columns=self.feature_mapping)
        return X_renamed



class PreprocessFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer for preprocessing features by replacing missing values (0) in specified features with NaN
    to prepare them for imputation.

    Parameters:
    features_no_measurements: List of feature column names where 0 represents missing values.
    """
    def __init__(self, features_no_measurements: List[str]) -> None:
        self.features_no_measurements = features_no_measurements

    def fit(self, X: pd.DataFrame, y : Optional[pd.DataFrame] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces 0 values in the specified columns with NaN.
        
        Parameters:
        - X: Input data (pandas DataFrame or numpy array) with features to be processed.
        
        Returns:
        - X: Transformed data where 0 values in specified columns are replaced with NaN.
        """
        X_nan = X.copy()
        X_nan[self.features_no_measurements] = X_nan[self.features_no_measurements].astype(float)
        X_nan.loc[:, self.features_no_measurements] = X_nan.loc[:, self.features_no_measurements].replace(0, np.nan)
        return X_nan


class KNNImputationByGroup(BaseEstimator, TransformerMixin):
    """
    Custom transformer for KNN imputation on feature groups determined with PCA.
    This performs imputation using KNN on predefined groups of features.

    Parameters:
    columns: List of column names in the DataFrame to be processed.
    n_neighbors: Number of neighbors to use for imputation.
    weights: Weighting function used in KNN ("uniform" or "distance").
    """
    def __init__(self, columns: List[str], n_neighbors: int = 5, weights: str = "uniform") -> None:
        self.columns = columns
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputers = []

        # Hard-coded feature groups (based on domain knowledge or PCA)
        self.groups = [
            ["BP", "Glucose", "Insulin"],  # Group 1
            ["BMI", "DPF", "Skin"]         # Group 2
        ]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KNNImputationByGroup':
        """
        Fits the KNN imputer to each feature group.

        Parameters:
        - X: Input data (numpy array) with features.
        - y: Target labels (not used here).

        Returns:
        - self: The fitted transformer.
        """
        self.imputers = []
        for group in self.groups:
            # Get indices of columns for the current group
            group_indices = [self.columns.index(col) for col in group]
            
            knn_imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
            knn_imputer.fit(X[:, group_indices])  # Fit the KNN imputer to each group of columns
            self.imputers.append(knn_imputer)
        
        return self

    def transform(self,  X: np.ndarray) -> np.ndarray:
        """
        Applies KNN imputation to the feature groups.
        
        Parameters:
        - X: Input data (pandas DataFrame or numpy array) with missing values to be imputed.

        Returns:
        - X_imputed: Transformed data with missing values imputed for each group.
        """
        X_imputed = X.copy()
        for group, imputer in zip(self.groups, self.imputers):
            # Get indices of columns for the current group
            group_indices = [self.columns.index(col) for col in group]
            
            X_imputed[:, group_indices] = imputer.transform(X[:, group_indices])  # Transform each group
            
        return X_imputed


class InverseScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for reversing the scaling of features to their original scale (only used for visualization).
    
    Parameters:
    - scaler: The scaler used for the initial scaling (e.g., StandardScaler).
    """
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'InverseScaler':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transforms the scaled data to return the features in their original scale.
        
        Parameters:
        - X: Input data (scaled features).
        
        Returns:
        - X: Inverse-transformed data.
        """
        return self.scaler.inverse_transform(X)


class AddKMeansClusterFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer for applying KMeans clustering and adding the resulting cluster label as a new feature.
    
    Parameters:
    - features (List[str]): List of feature column names to be used for KMeans clustering.
    - k (int): Number of clusters for KMeans.
    """
    def __init__(self, features: List[str], k: int = 2):
        self.k = k
        self.kmeans = None
        self.features = features
    
    def fit(self, X: np.ndarray, y=None):
        """
        Fits the KMeans model to the input features.

        Parameters:
        - X: The input features (numpy array).
        - y: Target labels (not used here).
        
        Returns:
        - self: The fitted transformer.
        """
        X = pd.DataFrame(X, columns=self.features)
        
        # Fit KMeans clustering on the specified features
        self.kmeans = KMeans(n_clusters=self.k, random_state=42)
        self.kmeans.fit(X[self.features])  # Fit only on the specified features
        return self
    
    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """
        Applies KMeans clustering and appends the cluster labels as a new feature.

        Parameters:
        - X: The input features (pandas DataFrame or numpy array).
        
        Returns:
        - X: Transformed data with the new "Cluster" feature, always returned as a pandas DataFrame.
        """
        # If X is a numpy array, convert it to DataFrame for easier column handling

        X = pd.DataFrame(X, columns=self.features)
        cluster_labels = self.kmeans.predict(X[self.features])
        X["Cluster"] = cluster_labels
        
        return X
