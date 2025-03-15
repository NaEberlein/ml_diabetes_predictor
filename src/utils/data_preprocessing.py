import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Union
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def standardize_apply_pca(X: pd.DataFrame, n_components: Union[float, int] = 0.95) -> Tuple[np.ndarray, PCA]:
    """
    Standardize the data and apply PCA transformation.

    Parameters:
    X (pd.DataFrame): The input data to be processed.
    n_components (float or int): Number of components to keep. If float, it is the explained variance ratio.

    Returns:
    X_pca (np.ndarray): The transformed data after PCA.
    pca (PCA): The PCA object that was fitted.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca


def perform_kmeans(X: pd.DataFrame, k_values: range) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform KMeans clustering for different values of k and calculate silhouette scores.

    Parameters:
    X (pd.DataFrame): The data to cluster.
    k_values (range): The range of k values for KMeans clustering.

    Returns:
    kmeans_labels (list): List of KMeans labels for each value of k.
    silhouette_scores (list): Silhouette scores for each value of k.
    """
    kmeans_labels = []
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        kmeans_labels.append(kmeans.labels_)
        sil_score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(sil_score)
        
    return kmeans_labels, silhouette_scores
    
def impute_and_extract_missing(data: pd.DataFrame, features: list[str], target: str, pipeline_name: str,
                              pipelines_dict: Dict[str, Pipeline], missing_values_mask: pd.DataFrame) -> pd.DataFrame:
    """
    Apply an imputation pipeline and return only the imputed rows.

    Parameters:
    data (pd.DataFrame): The original dataset.
    features (list[str]): Feature column names.
    target (str): Target column name.
    pipeline_name (str): The imputation method name (must exist in pipelines_dict).
    pipelines_dict (dict[str, Pipeline]): Dictionary of pre-defined pipelines.
    missing_values_mask (pd.DataFrame): Boolean mask identifying missing values.

    Returns:
    pd.DataFrame: A DataFrame with imputed data and not imputed data set to nan and an "Imputation_Method" column.
    """
    if pipeline_name not in pipelines_dict:
        raise ValueError(f"Pipeline '{pipeline_name}' not found in pipelines_dict.")

    X = data[features]
    y = data[target]

    pipeline = pipelines_dict[pipeline_name]
    X_imputed = pipeline.fit_transform(X)

    data_imputed = pd.DataFrame(X_imputed, columns=features)
    data_imputed[target] = y

    # Set non imputed values to nan
    data_imputed = data_imputed[missing_values_mask]

    # Mark imputation method
    data_imputed["Imputation_Method"] = pipeline_name

    return data_imputed


def create_artificial_missing_data(data: pd.DataFrame, random_state: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Introduce artificial missing values into a dataset that originally has no missing values.
    The missing value fractions are derived from the proportion of missing values in the original dataset data.

    This allows for controlled evaluation of imputation techniques.
    
    Parameters:
    data (pd.DataFrame): The original dataset.
    random_state (int or None): Random seed for reproducibility.

    Returns:
    data_no_missing: A subset of the dataset with no missing values.  
    data_with_missing: The same subset, but with artificially introduced missing values.
    """
    data_no_missing = data.dropna().reset_index(drop=True) # only use rows with no missing values

    nan_count = data.isna().sum()
    total_count = len(data)
    nan_ratio = nan_count / total_count
    features_missing_fraction = nan_ratio.to_dict()

    data_with_missing = introduce_missing_values(data_no_missing, features_missing_fraction, random_state=random_state)
    data_with_missing = data_with_missing.reset_index(drop=True)
    
    return data_no_missing, data_with_missing


def introduce_missing_values(data: pd.DataFrame, features_missing_fraction: Dict[str, float],
                                       random_state: Union[int, None] = None) -> pd.DataFrame:
    """
    In random rows introduce missing values in the specified columns.

    Parameters:
    data (pd.DataFrame): The dataset in which missing values will be inserted.
    features_missing_fraction (dict[str, float]): A dictionary where keys are feature names and values are the fractions of missing values.
    random_state (int or None): Random seed for reproducibility.

    Returns:
    data_nan (pd.DataFrame): The dataset with missing values.
    """
    rng = np.random.RandomState(random_state)
    data_nan = data.copy()
    for feature, missing_fraction in features_missing_fraction.items():
        num_missing = int(len(data_nan) * missing_fraction)
        missing_indices = rng.choice(data_nan.index.to_numpy(), size=num_missing, replace=False)
        data_nan.loc[missing_indices, feature] = np.nan
    
    return data_nan


def compute_rmse(original_values: pd.DataFrame, imputed_values: pd.DataFrame, missing_mask: pd.DataFrame) -> float:
    """
    Computes the RMSE for the imputed data compared to the original data.
    
    Parameters:
    original_values (pd.DataFrame): The original data with no missing values.
    imputed_values (pd.DataFrame): The imputed data with missing values filled.
    missing_mask (pd.DataFrame): A mask indicating where the missing values were.
    
    Returns:
    rmse (float): The computed RMSE value.
    """
    # Only consider the missing values using the mask
    original_values = original_values[missing_mask].replace(np.nan, 0) # need to replace nan with 0 to prevent error
    imputed_values = imputed_values[missing_mask].replace(np.nan, 0) # need to replace nan with 0 to prevent error

    return np.sqrt(mean_squared_error(original_values, imputed_values))


def run_pipeline_for_neighbors(data: pd.DataFrame, pipelines_dict: dict, n_neighbors_range: range, runs: int = 9) -> dict:
    """
    Run different imputation methods muliple times for different values of n_neighbors and calculate RMSE
    between original values and imputed values for each pipeline.
    
    Parameters:
    data (pd.DataFrame): The data to perform imputation on.
    pipelines_dict (dict): Dictionary of pipeline names and pipeline objects.
    n_neighbors_range (range): The range of n_neighbors values to test.
    runs (int): Number of times to run with different random states.
    
    Returns:
    rmse_results (dict): A dictionary containing RMSE values for each pipeline.
    """
    rmse_results = {}

    # Loop over the values of n_neighbors
    for n in n_neighbors_range:
        for i in range(runs):  # Run the process multiple times
            # Create missing values with different random states
            data_no_missing, data_with_missing = create_artificial_missing_data(data, random_state=i)

            # Create a mask to track where the missing values are
            missing_mask = data_with_missing.isna()

            # Loop through each pipeline in the pipelines dictionary
            for pipeline_name, pipeline in pipelines_dict.items():
                # If the pipeline uses KNN, set the current value of n_neighbors
                if "knn" in pipeline_name:
                    pipeline.set_params(imputer__n_neighbors=n)

                # Fit the pipeline to the data with missing values
                data_imputed = pipeline.fit_transform(data_with_missing)

                # Convert the resulting array back to a DataFrame
                data_imputed = pd.DataFrame(data_imputed, columns=data_with_missing.columns)

                # Compute RMSE
                rmse = compute_rmse(data_no_missing, data_imputed, missing_mask)

                # If the pipeline name is not in the results dictionary, initialize it
                if pipeline_name not in rmse_results:
                    rmse_results[pipeline_name] = np.zeros((len(n_neighbors_range), runs))

                # Store the RMSE value in the correct position
                rmse_results[pipeline_name][n - 1, i] = rmse

    return rmse_results


def generate_imputed_datasets(pipelines_dict, data, features, target):
    """
    Apply different imputation pipelines and return a dictionary of imputed datasets.
    
    Parameters:
    pipelines_dict: Dictionary containing imputation pipelines.
    data: The original dataset.
    features: List of features to be imputed.
    target: The target variable.
    
    Returns:
    data_imputed_dict: Dictionary with pipeline names as keys and the corresponding imputed datasets as values.
    """
    data_imputed_dict = {}
    for pipeline_name, pipeline in pipelines_dict.items():
        data_imputed = pipeline.fit_transform(data[features])
        data_imputed = pd.DataFrame(data_imputed, columns=features)
        data_imputed[target] = data[target]  # Add the target variable back to the imputed data
        data_imputed_dict[pipeline_name] = data_imputed
    return data_imputed_dict

