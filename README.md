## Disease Prediction from Medical Data

## Project Overview
This project focuses on building and evaluating machine learning models for disease prediction using medical data. Specifically, it demonstrates a classification task to differentiate between benign and malignant breast cancer tumors. The aim is to provide a practical, beginner-level guide to implementing a machine learning pipeline, from data loading and preprocessing to model training, evaluation, and comparison.

## Goal
The primary goal of this project is to develop and compare classification models (Logistic Regression and Random Forest Classifier) to accurately predict disease outcomes from medical data. The project emphasizes practical implementation over deep theoretical dives, providing a clear workflow for medical data analysis.

## Dataset
This project utilizes the Breast Cancer Wisconsin (Diagnostic) Dataset, which is readily available through sklearn.datasets. The dataset comprises features computed from digitized images of breast mass fine-needle aspirates (FNAs).

## Key Characteristics
Features (X): 30 numerical features representing characteristics of cell nuclei (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, etc.).
Target (y): A binary variable indicating the diagnosis (0 for Malignant, 1 for Benign).
Size: 569 samples with no missing values.

## Methodology
The project follows a standard machine learning methodology:

Data Loading: The Breast Cancer Wisconsin dataset is loaded.
Initial Data Exploration: Basic checks are performed to understand the dataset's structure, data types, and confirm the absence of missing values.

Data Preprocessing: The dataset is split into training (80%) and testing (20%) sets using train_test_split, with stratification to maintain class proportions.
Numerical features are scaled using StandardScaler to normalize their range, applied after the split to prevent data leakage.

Model Training & Evaluation
Logistic Regression: A LogisticRegression model is trained on the scaled training data.
Random Forest Classifier: A RandomForestClassifier is trained on the scaled training data.
Both models are evaluated using accuracy_score, classification_report, and confusion_matrix on the test set.
Model Comparison: The performance of both models is compared based on accuracy and detailed classification metrics, followed by a visualization of their accuracies.

## Key Findings
Model	Accuracy	False Positives	False Negatives
Logistic Regression	98.25%	1	1
Random Forest	95.61%	3	2
Logistic Regression achieved a higher accuracy (98.25%) and showed slightly better performance with fewer misclassifications compared to the Random Forest Classifier on this dataset.
Both models demonstrated strong predictive capabilities, indicating that the features extracted from the medical images are highly relevant for diagnosis.

## Tools Used
Python
pandas: For data manipulation and analysis.
numpy: For numerical operations.
scikit-learn: For dataset loading, data preprocessing (scaling, splitting), and implementing machine learning models (Logistic Regression, Random Forest).
matplotlib & seaborn: For data visualization.

## Conclusion
This project successfully implemented and evaluated two classification models for breast cancer prediction. Logistic Regression emerged as the slightly superior model for this particular dataset, offering a high level of accuracy and reliable predictions. The workflow demonstrated here provides a solid foundation for approaching similar disease prediction tasks using machine learning.
