#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2025
@author: robertgulmann
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =========================
# Step 1: Final Results
# =========================

results = {
    'Model': [
        'LogReg - Original', 'LogReg - Capped',
        'Decision Tree - Original', 'Decision Tree - Capped',
        'KNN - Original', 'KNN - Capped',
        'LogReg PCA - Original', 'LogReg PCA - Capped',
        'Decision Tree PCA - Original', 'Decision Tree PCA - Capped',
        'KNN PCA - Original', 'KNN PCA - Capped'
    ],
    'Accuracy': [
        0.7869, 0.7705, 0.7541, 0.7541, 0.8033, 0.8033,
        0.7705, 0.7705, 0.6885, 0.7377, 0.7869, 0.7869
    ],
    'AUC': [
        0.8728, 0.8761, 0.7689, 0.7689, 0.8831, 0.8750,
        0.8642, 0.8642, 0.6918, 0.7403, 0.8664, 0.8696
    ]
}

df_results = pd.DataFrame(results)

# =========================
# Step 2: Bar Chart (Visual 1)
# =========================

x = np.arange(len(df_results['Model']))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))
rects1 = ax.bar(x - width/2, df_results['Accuracy'], width, label='Accuracy')
rects2 = ax.bar(x + width/2, df_results['AUC'], width, label='AUC')

ax.set_xticks(x)
ax.set_xticklabels(df_results['Model'], rotation=60, ha='right')
ax.set_ylim(0.6, 1)
ax.set_ylabel('Score')
ax.set_title('Comparison of Accuracy and AUC Across All Model Variants')
ax.legend(title="Metric", loc="upper left")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with values
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', fontsize=8)

plt.tight_layout()
plt.show()

# =========================
# Step 3: Confusion Matrix (Visual 2)
# =========================

try:
    y_test_orig
    y_pred_orig
except NameError:
    print("Generating y_test_orig and y_pred_orig using KNN on original dataset...")
    df_orig = pd.read_csv('/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python /HeartDisease_df_original.csv')
    X = df_orig.drop("target", axis=1)
    y = df_orig["target"]
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_orig, y_train_orig)
    y_pred_orig = knn.predict(X_test_orig)

cm = confusion_matrix(y_test_orig, y_pred_orig)
class_names = ['No Disease', 'Disease']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title("Confusion Matrix: KNN on Original Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# =========================
# Step 4: ROC Curves (Visual 3)
# =========================

try:
    # Models and their probabilities
    model_probs = {
        'LogReg': y_proba_lr,
        'KNN': y_proba_knn,
        'Decision Tree': y_proba_dt,
        'LogReg PCA': y_proba_lr_pca,
        'KNN PCA': y_proba_knn_pca,
        'Decision Tree PCA': y_proba_dt_pca
    }

    plt.figure(figsize=(10, 7))

    for label, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test_orig, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

except NameError:
    print("⚠️ Predicted probabilities (e.g., y_proba_lr, y_proba_knn) are missing. Load them from your model scripts to show ROC curves.")