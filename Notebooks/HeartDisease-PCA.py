#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:35:32 2025

@author: robertgulmann
"""

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

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

# Step 5: Apply PCA (Keep 95% of variance)
pca = PCA(n_components=0.95)

X_train_orig_pca = pca.fit_transform(X_train_orig)
X_test_orig_pca = pca.transform(X_test_orig)

X_train_cap_pca = pca.fit_transform(X_train_cap)
X_test_cap_pca = pca.transform(X_test_cap)

print(f"\nNumber of PCA Components (Original Data): {X_train_orig_pca.shape[1]}")
print(f"Number of PCA Components (Capped Data): {X_train_cap_pca.shape[1]}")

# Step 6: Train & Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

print("\nTraining models on PCA-transformed datasets...")

for name, model in models.items():
    print(f"\nTraining {name} on PCA-Transformed Original Data...")
    model.fit(X_train_orig_pca, y_train_orig)
    y_pred_orig_pca = model.predict(X_test_orig_pca)
    acc_orig_pca = accuracy_score(y_test_orig, y_pred_orig_pca)
    prob_orig_pca = model.predict_proba(X_test_orig_pca)[:, 1]
    auc_orig = roc_auc_score(y_test_orig, prob_orig_pca)
    fpr_orig, tpr_orig, _ = roc_curve(y_test_orig, prob_orig_pca)

    print(f"{name} Accuracy on PCA-Transformed Original Data: {acc_orig_pca:.4f}")
    print(f"{name} AUC on PCA-Transformed Original Data: {auc_orig:.4f}")
    print(classification_report(y_test_orig, y_pred_orig_pca))

    print(f"\nTraining {name} on PCA-Transformed Capped Data...")
    model.fit(X_train_cap_pca, y_train_cap)
    y_pred_cap_pca = model.predict(X_test_cap_pca)
    acc_cap_pca = accuracy_score(y_test_cap, y_pred_cap_pca)
    prob_cap_pca = model.predict_proba(X_test_cap_pca)[:, 1]
    auc_cap = roc_auc_score(y_test_cap, prob_cap_pca)
    fpr_cap, tpr_cap, _ = roc_curve(y_test_cap, prob_cap_pca)

    print(f"{name} Accuracy on PCA-Transformed Capped Data: {acc_cap_pca:.4f}")
    print(f"{name} AUC on PCA-Transformed Capped Data: {auc_cap:.4f}")
    print(classification_report(y_test_cap, y_pred_cap_pca))

    results[name] = {
        "acc_orig": acc_orig_pca,
        "acc_cap": acc_cap_pca,
        "auc_orig": auc_orig,
        "auc_cap": auc_cap,
        "fpr_orig": fpr_orig,
        "tpr_orig": tpr_orig,
        "fpr_cap": fpr_cap,
        "tpr_cap": tpr_cap,
    }

# Step 7: Compare Accuracy and AUC
print("\nFinal Accuracy and AUC Comparison (PCA Models):")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  - Accuracy (Original) = {metrics['acc_orig']:.4f}")
    print(f"  - Accuracy (Capped)   = {metrics['acc_cap']:.4f}")
    print(f"  - AUC (Original)      = {metrics['auc_orig']:.4f}")
    print(f"  - AUC (Capped)        = {metrics['auc_cap']:.4f}")

# Step 8: Plot ROC curves
plt.figure(figsize=(10, 7))
for model, metrics in results.items():
    plt.plot(metrics['fpr_orig'], metrics['tpr_orig'], linestyle='--', label=f"{model} (Orig, AUC={metrics['auc_orig']:.2f})")
    plt.plot(metrics['fpr_cap'], metrics['tpr_cap'], linestyle='-', label=f"{model} (Capped, AUC={metrics['auc_cap']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for PCA-Transformed Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()