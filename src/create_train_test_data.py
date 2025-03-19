"""
This script creates a train and test data from the Pima Indians Diabetes dataset by performing the following tasks:

1. Loads the original dataset from a CSV file.
2. Checks if the data is valid, ensuring the correct number of rows (768) and columns, and that the necessary columns are present.
3. Splits the dataset into training and testing sets (default split: 70% train, 30% test).
4. Saves the training and testing sets to separate CSV files if they don't already exist.


Notes:
- The dataset should have 768 rows and the columns 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', and 'Outcome'.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): Path to the CSV file to be loaded.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the CSV is empty or has incorrect number of rows/columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
       
    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        # Use only the file name, not the full path
        file_name = os.path.basename(file_path)
        raise ValueError(f"The CSV file {file_name} is empty.")

    
    # Check if the DataFrame has the correct number of rows (768 rows)
    if len(data) != 768:
        raise ValueError(f"The CSV file {file_path} does not have 768 rows. It has {len(data)} rows.")
    
    # Check if the required columns are present
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                        'DiabetesPedigreeFunction', 'Age', 'Outcome']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the CSV file: {', '.join(missing_columns)}")
    
    return data

def save_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Parameters:
    data (pd.DataFrame): The DataFrame to be saved.
    file_path (str): The path where the CSV file will be saved.
    """
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}.")


def create_test_train_data(input_file_path: str, output_train_file_path: str,
                           output_test_file_path: str, test_size: float = 0.3, random_state: int = 42) -> None:
    """
    Splits a dataset into training and testing sets, and saves them as CSV files.

    Parameters:
    input_file_path (str): Path to the input CSV file containing the data.
    output_train_file_path (str): Path where the training data CSV file will be saved.
    output_test_file_path (str): Path where the testing data CSV file will be saved.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.3.
    random_state (int): Seed for random number generation. Default is 42.
    """
    # Load data
    data = load_csv(input_file_path)

    # Split into train and test data
    target = "Outcome"
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                        random_state=random_state)

    # Combine X_train with y_train and X_test with y_test
    data_train = pd.DataFrame(X_train, columns=features)
    data_train[target] = y_train
    data_test = pd.DataFrame(X_test, columns=features)
    data_test[target] = y_test

    # Save the train and test data (if not already exists)
    save_csv(data_train, output_train_file_path)
    save_csv(data_test, output_test_file_path)


if __name__ == "__main__":
    input_file = "../data/diabetes.csv"
    output_train_file = "../data/preprocessed_diabetes_train.csv"
    output_test_file = "../data/preprocessed_diabetes_test.csv"

    create_test_train_data(input_file, output_train_file, output_test_file)
