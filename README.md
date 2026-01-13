# Employee-Performance-Prediction
Overview
This project investigates the challenge of predicting employee performance scores (1-5) using a synthetic HR dataset of 100,000 records. The core objective is to build a classification model that remains fair and accurate even when low-performing employees are a small minority. The analysis rigorously tests the impact of class imbalance techniques and exposes the critical role of "Monthly Salary" as a predictive feature, highlighting potential data leakage issues common in HR analytics.

Key Features & Methodology
1. Data Processing and Exploratory Analysis

Dataset: 100,000 employee records with features including Demographics, Job Attributes, Productivity metrics, and Compensation.

EDA: Analyzed trends between workload (Projects Handled, Work Hours) and performance, revealing that while workload shows mild correlation, compensation is the strongest differentiator.

Preprocessing: Implemented Label Encoding for categorical variables, standardization for numeric features, and removed redundant features based on high multicollinearity (e.g., Hire Date vs. Years At Company).

2. Realistic Simulation of Class Imbalance

Problem: The original dataset was perfectly balanced, which is unrealistic for corporate environments.

Simulation: Engineered a custom imbalance scenario where Class 1 (lowest performers) constitutes only ~0.6% and Class 2 only ~6.2% of the data, mimicking real-world scarcity.

3. Model Development and Evaluation

Baseline: Logistic Regression models failed to detect minority classes without intervention, achieving 0.0 recall for Class 1.

Imbalance Handling: Evaluated multiple strategies:

Random Undersampling & SMOTE: Provided limited improvements.

Class Weighting: Applied to Logistic Regression, Random Forest, and XGBoost.

Custom Implementation: Built a Multi-class Logistic Regression from scratch with L2 regularization and gradient checking to understand the underlying mechanics .

4. Advanced Tree-Based Models

Random Forest: Utilized class_weight='balanced' to improve minority class detection.

XGBoost: Implemented a class-weighted XGBoost model.

Hyperparameter Tuning: Conducted Grid Search on XGBoost (tuning depth, learning rate, estimators), which emerged as the champion model.

Results
Champion Model: The tuned, class-weighted XGBoost model achieved a Macro F1-Score of 0.924 and 98.7% Accuracy.

Minority Class Success: Achieved a recall of 0.66 for Class 1 (the rarest class) and 0.85 for Class 2, significantly outperforming baseline models.

Critical Insight: The study proved that removing Monthly_Salary causes model performance to collapse (F1 drops to ~0.19), confirming it as the primary driver of prediction and a source of target leakage in this synthetic dataset.

Technologies Used
Python: Core programming language.

Libraries: Pandas, NumPy, Matplotlib, Seaborn (Data Manipulation & Visualization).

Machine Learning: Scikit-Learn (Logistic Regression, Random Forest), XGBoost (Gradient Boosting), Imbalanced-Learn (SMOTE, RandomUnderSampler).
