
# Diabetes Prediction Based on the Pima Indians Dataset

## Overview

This project aims to predict the likelihood of a person being diabetic based on diagnostic information. The dataset used comes from Kaggle, containing measurements from female patients over the age of 21.

## Goal

The goal is to develop machine learning models that predict whether a patient has diabetes based on their medical attributes.

---

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

- **Random Forest** 
- **Logistic Regression**
---
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
---
## Usage

1. **Download the dataset** from Kaggle and save it in the `data/` directory as `diabetes.csv`.

2. **Split the dataset** into training and test sets  with `src/create_train_test_data.py`
 
   **Note:** Only use the training data to prevent data leakage.

4. **Exploratory Data Analysis (EDA):**  
   Explore and visualise the data with `notebooks/01_data_exploration.ipynb`.

   
5. **Hyperparameter Tuning:** 
   Pipelines are defined based on insights from the EDA in

   - Random Forest `src/pipelines/define_pipelines_random_forest.py`
   - Logistic Regression `src/pipelines/define_pipelines_log_regression.py`

   Run hyperparameter tuning with Grid Search to optimize the model's performance with scripts in  `python src/grid_searches/`
   
7. **Evaluate the Results:**  
   Explore and evaluate the results from the hyperparameter tuning in the notebooks

   `notebooks/02a_model_evaluation_random_forest.ipynb` and `notebooks/02b_model_evaluation_log_regression.ipynb`
   
   The best-performing model for each classifier is saved as a serialized pipeline in `src/models/`.
---
### Best Classifiers:
**Random Forest**:
| Data       | Accuracy     | F1 Score     | ROC AUC   | Precision | Recall  |
|------------|--------------|--------------|-----------|-----------|---------|
| **Validation** | 0.7711 ± 0.0454 | 0.6956 ± 0.0575 | 0.8425 ± 0.0377 | --        | --      |
| **Test**       | 0.7229        | 0.6484        | 0.8412    | 0.5842    | 0.7284  |
| **Train**      | 0.7803        | 0.7108        | 0.8802    | 0.6561    | 0.7754  |


**Logistic Regression**
| Data       | Accuracy     | F1 Score     | ROC AUC   | Precision | Recall  |
|------------|--------------|--------------|-----------|-----------|---------|
| **Validation** | 0.7582 ± 0.0410 | 0.6706 ± 0.0550 | 0.8420 ± 0.0418 | --        | --      |
| **Test**       | 0.7446        | 0.6509        | 0.8356    | 0.625     | 0.679   |
| **Train**      | 0.7728        | 0.6935        | 0.8483    | 0.654     | 0.738   |

Note: The metrics are based on the best hyperparameters found through 10-fold cross-validation. Precision and Recall are calculated based on the positive class (diabetic).

---

### Best Classifiers:

#### **Random Forest**:

| Data       | Accuracy     | F1 Score     | ROC AUC   | Precision | Recall  |
|------------|--------------|--------------|-----------|-----------|---------|
| **Validation** | 0.7711 ± 0.0454 | 0.6956 ± 0.0575 | 0.8425 ± 0.0377 | --        | --      |
| **Test**       | 0.7229        | 0.6484        | 0.8412    | 0.5842    | 0.7284  |
| **Train**      | 0.7803        | 0.7108        | 0.8802    | 0.6561    | 0.7754  |

#### **Logistic Regression**:

| Data       | Accuracy     | F1 Score     | ROC AUC   | Precision | Recall  |
|------------|--------------|--------------|-----------|-----------|---------|
| **Validation** | 0.7582 ± 0.0410 | 0.6706 ± 0.0550 | 0.8420 ± 0.0418 | --        | --      |
| **Test**       | 0.7446        | 0.6509        | 0.8356    | 0.625     | 0.679   |
| **Train**      | 0.7728        | 0.6935        | 0.8483    | 0.654     | 0.738   |

---

### Discussion of the Results:

**Random Forest**:
- **Overfitting Concern**: The Random Forest model shows some signs of overfitting, with a noticeable gap between training performance and test/validation performance. This suggests the model may not generalize well to unseen data.
- **ROC AUC**: The model performs well in terms of ROC AUC, indicating it effectively distinguishes between diabetic and non-diabetic classes.
- **Recall**: The recall is higher than for the Logistic Regression, meaning Random Forest is better at identifying diabetic patients.

**Logistic Regression**:
- **Better Generalization**: The Logistic Regression shows more consistent performance across training, validation, and test sets. This suggests it generalizes better and doesn't overfit.
- **Accuracy and F1 Score**: Logistic Regression performs slightly better in terms of accuracy and F1 score on the test set but captures fewer diabetic individuals due to lower recall.

#### Ideas:
- Explore feature engineering techniques to help mitigate **Random Forest** overfitting.
- Experiment with other classifiers (f.e.**XGBoost**) for potential improvements.
--- 



## License

This project is licensed under the [MIT License](LICENSE.txt).
