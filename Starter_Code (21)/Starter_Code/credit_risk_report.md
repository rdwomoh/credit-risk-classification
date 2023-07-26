# Module 20 Credit Risk Report

## Overview of the Analysis

*Analysis Purpose: The main objective of this analysis is to assess whether the Logistic Regression machine learning model can effectively predict healthy loans and high-risk loans using either the original dataset or a resampled dataset to address class imbalance.

Dataset Description: The dataset employed for the analysis contains information on 77,536 loans. It encompasses various features such as loan_size, interest_rate, borrower_income, debt_to_income ratio, number_of_accounts, derogatory_marks, total_debt, and loan_status. The prediction target is the loan_status, and the remaining columns are utilized as features to train the model and make predictions.

Stages of the Machine Learning Process: The machine learning process consists of several well-defined stages that need to be followed in a prescribed order to ensure accurate model predictions. These stages are as follows:

1. Data Preparation: This stage involves importing the dataset, creating a DataFrame, and assessing the columns and features.

2. Data Separation: The data is divided into features and labels. The labels represent the loan_status, with values indicating healthy (0) or high-risk (1) loans. The remaining data constitutes the features used for training and testing the model.

3. Train-Test Split: The features and labels data are further split into separate training and testing datasets to evaluate the model's performance.

4. Model Import: The LogisticRegression model from SKLearn is imported for implementation in this analysis.

5. Model Instantiation: The model is instantiated, creating an instance of the LogisticRegression model.

6. Model Fitting: The model is trained using the training data to learn patterns and relationships within the features and labels.

7. Prediction: The model is used to make predictions on the test data, utilizing the extracted patterns.

8. Evaluation: The predictions are evaluated using various metrics, including the accuracy score, a confusion matrix, and a classification report.

Machine Learning Methods: The primary machine learning model utilized in this analysis is the LogisticRegression model from SKLearn. Supporting functions such as train_test_split from SKLearn are used to facilitate the process, while evaluation is carried out with the help of the confusion_matrix and classification_report functions from SKLearn.

## Results

* Machine Learning Model 1 - Logistic Regression:

  - Accuracy score: 0.99
  - Precision:
    - Class 0: 1.00
    - Class 1: 0.85
  - Recall:
    - Class 0: 0.99
    - Class 1: 0.91

## Summary

The Logistic Regression model exhibited excellent performance, particularly in predicting Class 0 (healthy loans), where both precision and recall were nearly perfect.

For Class 1 (high-risk loans), the model's precision was 0.85, indicating that it correctly identified 85% of the high-risk loans among the predicted positives. The recall for Class 1 was 0.91, meaning the model accurately detected 91% of the actual high-risk loans.

It's worth noting that the model had a higher number of false positives than false negatives for Class 1, as indicated by the precision score. However, the overall predictive ability for both classes was commendable, given the features used for training the data.

While the Logistic Regression model shows promise, further evaluation on different datasets would be necessary to validate its suitability for predicting the health status of loans effectively.