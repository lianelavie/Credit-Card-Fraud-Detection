# Credit-Card-Fraud-Detection
This project focuses on detecting fraudulent transactions using machine learning, with the help of the Credit Card Fraud Detection dataset. The primary goal is to build a classification model to accurately identify fraudulent credit card transactions.

# 📁 Dataset
Source: Kaggle Credit Card Fraud Detection Dataset

Size: 284,807 transactions

Features: 30 numerical features (V1–V28 are PCA transformed), Time, Amount

Target: Class (0 for valid, 1 for fraud)

# 🛠️ Libraries Used
python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# 📊 Exploratory Data Analysis (EDA)
Dataset Summary: 284,807 rows × 31 columns

Fraudulent Transactions: 492 (~0.17% of total)

Valid Transactions: 284,315

Imbalance Issue: Highly skewed target variable (Class)

Distribution of Amount for:
Fraudulent Transactions

Mean: 122.21

Max: 2125.87

Valid Transactions

Mean: 88.29

Max: 25,691.16

# Correlation Heatmap
A heatmap of feature correlations is generated to understand relationships among variables.

# ⚙️ Model Building
Model Used: Random Forest Classifier

Train/Test Split: 80/20

Target Variable: Class

Feature Variables: All except Class

# Code:
python
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)
📈 Evaluation Metrics
Metric	Value
Accuracy	99.96%
Precision	96.25%
Recall	78.57%
F1 Score	86.52%
Matthews Correlation Coef	0.869

Despite the data imbalance, the Random Forest model performs well with high precision and reasonable recall.

# 🔍 Confusion Matrix
A confusion matrix is plotted using Seaborn to visualize the prediction results:

X-axis: Predicted class

Y-axis: True class

# 📌 Conclusion
The Random Forest model provides high accuracy and precision.

There’s room to improve recall using techniques like:

SMOTE (Synthetic Minority Over-sampling Technique)

Anomaly Detection methods

Other classifiers like XGBoost or Isolation Forest

# 🧠 Future Work
Apply other machine learning models for comparison

Address data imbalance using resampling techniques

Use feature engineering to improve interpretability

Deploy the model with a simple web interface

#📝 Author
Created as part of a machine learning project for credit card fraud detection.




