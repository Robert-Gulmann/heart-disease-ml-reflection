#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:33:02 2025

@author: robertgulmann
"""


# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 2: Load the cleaned datasets
file_path_original = '/Users/robertgulmann/heart-disease-ml-reflection/Data/Heart-Disease-Dataset.csv'
file_path_capped = '/Users/robertgulmann/heart-disease-ml-reflection/Data/HeartDisease_df_capped.csv'

df_original = pd.read_csv(file_path_original)
df_capped = pd.read_csv(file_path_capped)

# Step 3: Define features (X) and target (y)
X_original = df_original.drop(columns=['target'])
y_original = df_original['target']

X_capped = df_capped.drop(columns=['target'])
y_capped = df_capped['target']

# Step 4: Split into train (80%) and test (20%) sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
X_train_cap, X_test_cap, y_train_cap, y_test_cap = train_test_split(X_capped, y_capped, test_size=0.2, random_state=42)

# Step 5: Set up hyperparameter tuning for Logistic Regression
param_grid = {'C': [0.01, 0.05, 0.1, 1, 10]}

grid_search_orig = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search_orig.fit(X_train_orig, y_train_orig)

grid_search_cap = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search_cap.fit(X_train_cap, y_train_cap)

# Step 6: Train best Logistic Regression models
best_log_reg_orig = grid_search_orig.best_estimator_
best_log_reg_cap = grid_search_cap.best_estimator_

y_pred_orig = best_log_reg_orig.predict(X_test_orig)
y_pred_cap = best_log_reg_cap.predict(X_test_cap)

# Evaluation for original dataset
precision_orig = precision_score(y_test_orig, y_pred_orig)
recall_orig = recall_score(y_test_orig, y_pred_orig)
f1_orig = f1_score(y_test_orig, y_pred_orig)
auc_orig = roc_auc_score(y_test_orig, best_log_reg_orig.predict_proba(X_test_orig)[:, 1])

print("\n🔹 Best Logistic Regression Model - Original Data:")
print(f"Training Accuracy: {best_log_reg_orig.score(X_train_orig, y_train_orig):.4f}")
print(f"Testing Accuracy: {best_log_reg_orig.score(X_test_orig, y_test_orig):.4f}")
print(f"Precision: {precision_orig:.4f}")
print(f"Recall: {recall_orig:.4f}")
print(f"F1 Score: {f1_orig:.4f}")
print(f"AUC Score: {auc_orig:.4f}")

# Evaluation for capped dataset
precision_cap = precision_score(y_test_cap, y_pred_cap)
recall_cap = recall_score(y_test_cap, y_pred_cap)
f1_cap = f1_score(y_test_cap, y_pred_cap)
auc_cap = roc_auc_score(y_test_cap, best_log_reg_cap.predict_proba(X_test_cap)[:, 1])

print("\n🔹 Best Logistic Regression Model - Capped Data:")
print(f"Training Accuracy: {best_log_reg_cap.score(X_train_cap, y_train_cap):.4f}")
print(f"Testing Accuracy: {best_log_reg_cap.score(X_test_cap, y_test_cap):.4f}")
print(f"Precision: {precision_cap:.4f}")
print(f"Recall: {recall_cap:.4f}")
print(f"F1 Score: {f1_cap:.4f}")
print(f"AUC Score: {auc_cap:.4f}")

# Plot ROC Curves
fpr_orig, tpr_orig, _ = roc_curve(y_test_orig, best_log_reg_orig.predict_proba(X_test_orig)[:, 1])
fpr_cap, tpr_cap, _ = roc_curve(y_test_cap, best_log_reg_cap.predict_proba(X_test_cap)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(fpr_orig, tpr_orig, label=f"Original Data (AUC = {auc_orig:.2f})")
plt.plot(fpr_cap, tpr_cap, label=f"Capped Data (AUC = {auc_cap:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()