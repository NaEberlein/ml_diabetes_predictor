import pytest
import pandas as pd
import os
from create_train_test_data import load_csv, save_csv, create_test_train_data

def generate_valid_data():
    """Generates valid mock data set with 768 rows."""
    data = {
        "Pregnancies": [1]*768,
        "Glucose": [85]*768,
        "BloodPressure": [66]*768,
        "SkinThickness": [29]*768,
        "Insulin": [0]*768,
        "BMI": [26.6]*768,
        "DiabetesPedigreeFunction": [0.351]*768,
        "Age": [31]*768,
        "Outcome": [0]*768
    }
    return pd.DataFrame(data)

def generate_invalid_data_columns():
    """Generates invalid mock data set with 768 rows, but without the column DiabetesPedigreeFunction"""
    data = {
        "Pregnancies": [1]*768,
        "Glucose": [85]*768,
        "BloodPressure": [66]*768,
        "SkinThickness": [29]*768,
        "Insulin": [0]*768,
        "BMI": [26.6]*768,
        "Age": [31]*768,
        "Outcome": [0]*768
    }
    return pd.DataFrame(data)

def generate_invalid_data_rows():
    """Generates invalid mock data set with 1 row."""
    data = {
        "Pregnancies": [1],
        "Glucose": [85],
        "BloodPressure": [66],
        "SkinThickness": [29],
        "Insulin": [0],
        "BMI": [26.6],
        "DiabetesPedigreeFunction": [0.351],
        "Age": [31],
        "Outcome": [0]
    }
    return pd.DataFrame(data)


mock_file = "mock_data.csv"

@pytest.fixture
def setup_valid_mock_csv(tmpdir):
    """Create temp csv file with 768 rows for test."""
    valid_data = generate_valid_data()
    file_path = tmpdir.join(mock_file)  # create path for tmp file in tmpdir
    valid_data.to_csv(file_path, index=False)
    yield str(file_path)  # return path of tmp file (tmp file deleted after test finished)

@pytest.fixture
def setup_invalid_columns_mock_csv(tmpdir):
    """Creates temp csv file with 768 rows for test, but with a missing column."""
    invalid_data = generate_invalid_data_columns()
    file_path = tmpdir.join(mock_file)  # create path for tmp file in tmpdir
    invalid_data.to_csv(file_path, index=False)
    yield str(file_path)  # return path of tmp file (tmp file deleted after test finished)

@pytest.fixture
def setup_invalid_rows_mock_csv(tmpdir):
    """Creates temp csv file with 1 row for test."""
    invalid_data = generate_invalid_data_rows()
    file_path = tmpdir.join(mock_file)  # create path for tmp file in tmpdir
    invalid_data.to_csv(file_path, index=False)
    yield str(file_path)  # return path of tmp file (tmp file deleted after test finished)


def test_load_csv_file_not_found():
    """Tests loading a non-existent CSV file."""
    invalid_path = "non_existent_file.csv"
    with pytest.raises(FileNotFoundError, match=f"The file {invalid_path} does not exist."):
        load_csv(invalid_path)


def test_load_csv_empty_file(tmpdir):
    """Tests loading an empty CSV file."""
    empty_file = tmpdir.join("mock_data.csv")
    empty_file.write("")  # Create an empty file
    with pytest.raises(pd.errors.EmptyDataError) as exc_info:
        load_csv(str(empty_file))


def test_load_csv_wrong_columns(setup_invalid_columns_mock_csv):
    """Tests loading a file with missing column."""
    with pytest.raises(ValueError) as exc_info:
        load_csv(setup_invalid_columns_mock_csv)
    assert "The following required columns are missing from the CSV file" in str(exc_info.value)
    assert "DiabetesPedigreeFunction" in str(exc_info.value)
    
def test_load_csv_wrong_rows(setup_invalid_rows_mock_csv):
    """Tests loading a file with wrong number of rows."""
    with pytest.raises(ValueError) as exc_info:
        load_csv(setup_invalid_rows_mock_csv)
    assert "does not have 768 rows" in str(exc_info.value)
    assert "It has 1 rows" in str(exc_info.value)

def test_load_csv_success(setup_valid_mock_csv):
    """Tests loading of file with correct number of rows"""
    df = load_csv(setup_valid_mock_csv)
    assert isinstance(df, pd.DataFrame) # checks that its a df
    assert df.shape[0] == 768  
    assert set(df.columns) == set(generate_valid_data().columns) # checks that it has the expected number of columns

def test_save_csv_creates_directory(tmpdir):
    """Tests saving a CSV ensures the directory is created if it does not exist."""
    new_dir = tmpdir.join("new_folder")  # create the directory first
    file_path = new_dir.join(mock_file)  # file inside the new folder
    print(new_dir)
    assert not os.path.exists(new_dir)  # initially, the file should not exist
    valid_data = generate_valid_data()
    save_csv(valid_data, str(file_path))  # save data to the file path
    assert os.path.exists(file_path)  # the file should now exist

def test_save_csv_overwrites_file(tmpdir):
    """Tests overwriting an existing file.""" 
    file_path = tmpdir.join(mock_file)
    valid_data = generate_valid_data()
    save_csv(valid_data, str(file_path))  # initially save
    original_df = pd.read_csv(str(file_path))
    assert original_df.equals(valid_data)  # validate initial save
    
    # Overwrite with new data (first 20 rows)
    valid_data_new = valid_data.head(20)  # keep only the first 20 rows
    save_csv(valid_data_new, str(file_path))  # overwrite with new data
    new_df = pd.read_csv(str(file_path))
    assert new_df.equals(valid_data_new)  # validate overwrite
