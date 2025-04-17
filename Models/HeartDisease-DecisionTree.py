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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 2: Load the cleaned datasets
file_path_original = "/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python /HeartDisease_df_original.csv"
file_path_capped = "/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python /HeartDisease_df_capped.csv"

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

# Evaluate best Decision Tree model for original dataset
print("\n🔹 Best Decision Tree Model - Original Data:")
print(f"Best Parameters: {grid_search_orig.best_params_}")
print(f"Training Accuracy: {best_tree_orig.score(X_train_orig, y_train_orig):.4f}")
print(f"Testing Accuracy: {best_tree_orig.score(X_test_orig, y_test_orig):.4f}")
print(classification_report(y_test_orig, y_pred_orig))

# Evaluate best Decision Tree model for capped dataset
print("\n🔹 Best Decision Tree Model - Capped Data:")
print(f"Best Parameters: {grid_search_cap.best_params_}")
print(f"Training Accuracy: {best_tree_cap.score(X_train_cap, y_train_cap):.4f}")
print(f"Testing Accuracy: {best_tree_cap.score(X_test_cap, y_test_cap):.4f}")
print(classification_report(y_test_cap, y_pred_cap))

# Step 7: ROC Curve and AUC Score

# Get prediction probabilities (needed for AUC)
y_prob_orig = best_tree_orig.predict_proba(X_test_orig)[:, 1]
y_prob_cap = best_tree_cap.predict_proba(X_test_cap)[:, 1]

# Compute ROC Curves
fpr_orig, tpr_orig, _ = roc_curve(y_test_orig, y_prob_orig)
fpr_cap, tpr_cap, _ = roc_curve(y_test_cap, y_prob_cap)

# Compute AUC Scores
auc_orig = roc_auc_score(y_test_orig, y_prob_orig)
auc_cap = roc_auc_score(y_test_cap, y_prob_cap)

print(f"\n🔹 AUC Score - Original Data: {auc_orig:.4f}")
print(f"🔹 AUC Score - Capped Data: {auc_cap:.4f}")

# Plot ROC Curves
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