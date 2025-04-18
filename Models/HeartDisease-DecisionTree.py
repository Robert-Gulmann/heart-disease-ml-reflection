#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:30:33 2025

@author: robertgulmann
"""

# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
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

# Step 5: Set up hyperparameter tuning for Decision Tree
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_split': [2, 5, 10, 15, 20]
}

grid_search_orig = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search_orig.fit(X_train_orig, y_train_orig)

grid_search_cap = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search_cap.fit(X_train_cap, y_train_cap)

# Step 6: Train best Decision Tree models
best_tree_orig = grid_search_orig.best_estimator_
best_tree_cap = grid_search_cap.best_estimator_

y_pred_orig = best_tree_orig.predict(X_test_orig)
y_pred_cap = best_tree_cap.predict(X_test_cap)

# Step 7: Evaluation Metrics
def evaluate_model(name, y_test, y_pred, y_proba, model, X_train, y_train, X_test):
    print(f"\n🔹 Best Decision Tree Model - {name}:")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Testing Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    return roc_curve(y_test, y_proba), roc_auc_score(y_test, y_proba)

# Probabilities for AUC
y_prob_orig = best_tree_orig.predict_proba(X_test_orig)[:, 1]
y_prob_cap = best_tree_cap.predict_proba(X_test_cap)[:, 1]

# Evaluate both models
(fpr_orig, tpr_orig, _), auc_orig = evaluate_model(
    "Original Data", y_test_orig, y_pred_orig, y_prob_orig, best_tree_orig, X_train_orig, y_train_orig, X_test_orig
)

(fpr_cap, tpr_cap, _), auc_cap = evaluate_model(
    "Capped Data", y_test_cap, y_pred_cap, y_prob_cap, best_tree_cap, X_train_cap, y_train_cap, X_test_cap
)

# Step 8: Plot ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_orig, tpr_orig, label=f"Original Data (AUC = {auc_orig:.2f})")
plt.plot(fpr_cap, tpr_cap, label=f"Capped Data (AUC = {auc_cap:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()