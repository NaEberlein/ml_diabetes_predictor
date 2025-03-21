# Notebooks Directory

This directory contains Jupyter notebooks used for data exploration and evaluation of various machine learning pipelines.

## Notebooks

### Data Exploration
`01_data_exploration.ipynb`
- **Purpose**: This notebook performs exploratory data analysis (EDA).
  
- **Key Steps**:
  - Overview of the dataset
  - Visualisation of key features
  - Handling missing values
  - Exploration of relationships between features and the target variable (PCA, correlations)
  - Imputation methods for missing values

### Model Evaluation
`02a_model_evaluation_random_forest.ipynb` and `02b_model_evaluation_log_regression.ipynb`
- **Purpose**: These notebooks evaluate the results of grid search for different pipelines, using Random Forest or Logistic Regression as classifiers.
  
- **Key Steps**:
  - Hyperparameter tuning via Grid Search
  - Evaluation metrics: Accuracy, F1 Score, ROC AUC
  - Model performance analysis on training, validation, and test sets
  - Visualisation of the best model's performance on the test data

---

## Model Evaluation

### 1. Evaluation of Hyperparameter Tuning Results
- The hyperparameter grid search for each model (Random Forest and Logistic Regression) is implemented with **10-fold stratified cross-validation**, where 70% of the dataset is used as training data.
- Multiple pipelines are tested, each with different preprocessing steps applied before the classifier.
- For each pipeline, the best model is selected based on:
  - **Validation metric scores** (accuracy, F1 score, and AUC)
  - **Difference between training and validation scores** (to assess overfitting)
  - **Standard deviation of the validation score** (to evaluate model stability)

#### Key Considerations:
- A **large gap** between training and validation scores indicates **overfitting** (high variance), meaning the model struggles to generalise well.
- A **high standard deviation** in validation scores suggests the model is **unstable** and may not perform consistently on new, unseen data.
- Both factors can significantly affect the model's final performance on the test data.
- The **best pipeline** is selected based on the above criteria.

---

### General Pipeline Setup

The pipeline involves several essential steps to ensure proper data processing, imputation, scaling, and model evaluation:

- **Preprocessing**: Replace `0` values with `NaN` to prepare for imputation.
- **Imputation**: Missing values are handled using the following methods:
  - **Simple Imputer** (mean, median)
  - **KMeans-based imputation**: Impute missing values using KMeans clustering, either using all features or only those with high correlation in PCA.
  
  **Note**: The order of imputation and scaling depends on the imputation method used:
  - **For KMeans-based imputation**: Standardisation (scaling) happens **before** imputation, as KMeans clustering works on scaled data.
  - **For Simple Imputation** (mean, median): Imputation happens **before** scaling to preserve the standardisation of the data.

- **Standard Scaler**: Standardise features (mean=0, std=1) after imputation, unless KMeans imputation is used.
- **Feature Addition**: Add KMeans clustering features to improve model performance.
- **Classifier**: The model is fine-tuned using grid search to select the best-performing parameters.

---

### Pipeline Overview

| #  | Imputation                                         | KMeans Feature Added | Pipeline Name                        |
|----|----------------------------------------------------|----------------------|--------------------------------------|
| 1  | KNN Imputation based on correlated features via PCA| ✔                    | KNN with PCA                         |
| 2  | KNN Imputation based on all features               | ✔                    | KNN                                  |
| 3  | Mean Imputation                                    | ✔                    | Mean Imputation                      |
| 4  | Median Imputation                                  | ✔                    | Median Imputation                    |
| 5  | None                                               | ✘                    | No Imputation & No KMeans            |
| 6  | KNN Imputation based on PCA                        | ✘                    | KNN with PCA & No KMeans             |
| 7  | KNN Imputation                                     | ✘                    | KNN & No KMeans                      |
| 8  | Mean Imputation                                    | ✘                    | Mean Imputation & No KMeans          |
| 9  | Median Imputation                                  | ✘                    | Median Imputation & No KMeans        |

- All other steps (e.g., preprocessing, standard scaler) are identical across pipelines. The Logistic Regression pipeline does not include the **None Imputation** option (cannot handle `NaN` values).

---

### Evaluation Metrics

- **Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  - May not be the best metric due to the class imbalance (class 0: 65%, class 1: 35%).

- **F1 Score**:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  - **Precision**: Measures how many predicted positive cases are actually positive.
  - **Recall**: Measures how many actual positive cases are correctly predicted.

- **AU ROC**:
  - Measures the model's ability to differentiate between classes.
  - AUC close to 1 indicates a good model; near 0.5 suggests random predictions.

---

### Model Selection Criteria

For each metric, the best model is selected based on the following criteria:

1. **Validation Metric Score**:
   - The best model is selected based on the highest validation metric score.
   - A model is considered eligible if:
     - The **difference** between the **train** and **validation** metrics is **less than 0.05**.
     - The **standard deviation** of the validation metric is **less than 0.05**.
   
   If none of the models meet these criteria, the model with the smallest difference between train and validation metrics (even if > 0.05) and the smallest standard deviation of the validation metric is selected.

2. **Model Selection**:
   - After applying the above criteria, the model with the **highest validation score** for each metric is chosen.
   - The **final model** is the one with the **highest overall validation score** across all metrics.

For each pipeline, model parameters are fine-tuned during a grid search process to select the optimal configuration for the best performance.

