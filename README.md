
# Diabetes Prediction Based on the Pima Indians Dataset

## Overview

This project aims to predict the likelihood of a person being diabetic based on diagnostic information. The dataset used comes from Kaggle, containing measurements from female patients over the age of 21.

## Goal

The goal is to develop machine learning models that predict whether a patient has diabetes based on their medical attributes.

## Dataset

**Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)

The dataset includes 768 patient records with the following variables:

### Target Variable:
- **Outcome:** A binary variable indicating whether the individual has diabetes:
  - `1`: The individual has diabetes.
  - `0`: The individual does not have diabetes.

### Feature Variables:
- **Pregnancies:** Number of pregnancies the patient has had.
- **Age:** Age of the patient in years.
- **Glucose:** Plasma glucose concentration during a 2-hour oral glucose tolerance test (mg/dL).
- **Blood Pressure:** Diastolic blood pressure (mmHg).
- **Skin Thickness:** Triceps skin fold thickness (mm).
- **Insulin:** 2-hour serum insulin concentration (mu U/ml).
- **BMI:** Body Mass Index (kg/m^2).
- **Diabetes Pedigree Function:** A function representing the likelihood of diabetes based on family history.

**Notes**
- Missing values are present in the following columns: `Glucose`, `Blood Pressure`, `Skin Thickness`, `Insulin`, and `BMI`, which are set to zero.

## Implemented Classifiers

- **Random Forest** (Additional classifiers will be added in future iterations)

## Installation

Follow these steps to set up this project locally:

1. Clone this repository:

   ```bash
   git clone https://github.com/NaEberlein/ml_diabetes_predictor.git
   cd ml_diabetes_predictor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the project package:

   ```bash
   pip install -e .
   ```

## Usage

1. **Download the dataset** from Kaggle and save it in the `data/` directory as `diabetes.csv`.

2. **Split the dataset** into training and test sets  with `src/create_train_test_data.py`
 
   **Note:** Only use the training data to prevent data leakage.

4. **Exploratory Data Analysis (EDA):**  
   Explore and visulaise the data with `notebooks/01_data_exploration.ipynb`.

   
5. **Hyperparameter Tuning:** 
   Pipelines are defined based on insights from the EDA in `src/pipelines/define_pipelines.py`

   Run hyperparameter tuning with Grid Search to optimize the model's performance with `python src/hyperparameter_tuning.py`
   
   The results of the tuning will be saved in `results/hyperparameter_tuning_results`

7. **Evaluate the Results:**  
   Explore and evaluate the results from the hyperparameter tuning in the notebook `notebooks/02_model_evaluation.ipynb`
   
   The best-performing model is saved as a serialized pipeline in `src/models/best_params/best_pipeline_model.pkl`


## Contribution

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m "Added new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE.txt).
