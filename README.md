
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
   
7. **Evaluate the Grid Search:**  

   `notebooks/02a_model_evaluation_random_forest.ipynb` and `notebooks/02b_model_evaluation_log_regression.ipynb`
   
   The best-performing model for each classifier is saved as a serialized pipeline in `src/models/`.

8. **Model Evalution on Test Data**:
   Evaluate the performance of the best random forest and logisitic regression model on the unseen test data
   in `notebooks/03_model_evaluation_on_test_data.ipynb`.
---

### Best Classifiers:

#### **Random Forest**:
| Data           | Accuracy        | F1 Score        | ROC AUC         | Precision       | Recall          |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| **Validation** | 0.7749 ± 0.0494 | 0.6976 ± 0.0625 | 0.8448 ± 0.0398 | 0.6667 ± 0.0866 | 0.7506 ± 0.1163 |
| **Test**       | 0.7489          | 0.6813          | 0.8454          | 0.6139          | 0.7654          |
| **Train**      | 0.8324          | 0.7805          | 0.9108          | 0.7175          | 0.8556          |

#### **Logistic Regression**:

| Data           | Accuracy        | F1 Score        | ROC AUC         | Precision       | Recall          |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| **Validation** | 0.7582 ± 0.0410 | 0.6706 ± 0.0550 | 0.8420 ± 0.0418 | 0.6451 ± 0.0655 | 0.7129 ± 0.1056 |
| **Test**       | 0.7662          | 0.6824          | 0.8365          | 0.6517          | 0.7160          |
| **Train**      | 0.7728          | 0.6904          | 0.8477          | 0.6570          | 0.7273          |

---

### Discussion of the Results:

**Random Forest**:
- **Overfitting Concern**: The Random Forest model shows signs of overfitting, with a noticeable gap between training performance and test/validation performance. This suggests the model may not generalize well to unseen data.
- **ROC AUC**: The model performs well in terms of ROC AUC, indicating it effectively distinguishes between diabetic and non-diabetic classes.
- **Recall**: The recall is higher than for the Logistic Regression, meaning Random Forest is better at identifying diabetic patients.

**Logistic Regression**:
- **Better Generalization**: The Logistic Regression shows more consistent performance across training, validation, and test sets.
- **Accuracy and F1 Score**: Logistic Regression performs slightly better in terms of accuracy and F1 score on the test set but captures fewer diabetic individuals due to lower recall.

#### Ideas:
- Explore feature engineering techniques to help mitigate **Random Forest** overfitting.
- Experiment with other classifiers (f.e.**XGBoost**) for potential improvements.
--- 


## License

This project is licensed under the [MIT License](LICENSE.txt).
