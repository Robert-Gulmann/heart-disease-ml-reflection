#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:14:41 2025

@author: robertgulmann
"""
import pandas as pd
import matplotlib.pyplot as plt

# Corrected paths with no space before the slash
original_path = '/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python/HeartDisease_df_original.csv'
capped_path = '/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python/HeartDisease_df_capped.csv'

df_orig = pd.read_csv(original_path)
df_capped = pd.read_csv(capped_path)

# Select relevant continuous features for visualisation
features_to_plot = ['chol', 'thalach', 'trestbps', 'oldpeak']

plt.figure(figsize=(12, 8))

for i, feature in enumerate(features_to_plot):
    plt.subplot(2, len(features_to_plot), i + 1)
    df_orig.boxplot(column=feature)
    plt.title(f'Original: {feature}')

    plt.subplot(2, len(features_to_plot), i + 1 + len(features_to_plot))
    df_capped.boxplot(column=feature)
    plt.title(f'Capped: {feature}')

plt.tight_layout()
plt.suptitle('Boxplots Before and After IQR Capping', y=1.02, fontsize=16)
plt.show()