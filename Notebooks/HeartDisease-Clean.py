#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:56:15 2025

@author: robertgulmann
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = '/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python /Heart-Disease-Dataset.csv'
df = pd.read_csv(file_path)

# Remove duplicates
df = df.drop_duplicates()

# Define categorical and numerical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Create two versions: original and capped
df_original = df.copy()
df_capped = df.copy()

# Function to cap outliers
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

# Apply capping to df_capped
for col in numerical_columns:
    cap_outliers(df_capped, col)

# Scale both datasets AFTER capping
scaler = StandardScaler()
df_original[numerical_columns] = scaler.fit_transform(df_original[numerical_columns])
df_capped[numerical_columns] = scaler.fit_transform(df_capped[numerical_columns])

# One-hot encode categorical columns AFTER scaling
encode_cols = ['cp', 'restecg', 'slope', 'thal']
df_original = pd.get_dummies(df_original, columns=encode_cols, drop_first=True)
df_capped = pd.get_dummies(df_capped, columns=encode_cols, drop_first=True)

# Compare boxplots for key features
melted_original = df_original[numerical_columns].copy()
melted_capped = df_capped[numerical_columns].copy()
melted_original["Dataset"] = "Original"
melted_capped["Dataset"] = "Capped"

melted_df = pd.concat([melted_original, melted_capped])
melted_df = pd.melt(melted_df, id_vars="Dataset", var_name="Feature", value_name="Value")

plt.figure(figsize=(14, 7))
sns.boxplot(x="Feature", y="Value", hue="Dataset", data=melted_df)
plt.title("Boxplots Before and After IQR Capping")
plt.legend(title="Dataset")
plt.tight_layout()
plt.show()

# Save processed datasets
save_path = "/Users/robertgulmann/Desktop/BS3/BAA1027–DataAnalytics–MachineLearning–Advanced–Python/"
df_original.to_csv(save_path + "HeartDisease_df_original.csv", index=False)
df_capped.to_csv(save_path + "HeartDisease_df_capped.csv", index=False)

print("Datasets saved successfully to:", save_path)