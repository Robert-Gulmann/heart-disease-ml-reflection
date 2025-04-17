#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:56:08 2025

@author: robertgulmann
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# File path to capped dataset (already scaled + encoded)
capped_path = '/Users/robertgulmann/Desktop/BS3/BAA1027-DataAnalytics-MachineLearning-Advanced-Python /HeartDisease_df_capped.csv'

# Load the capped dataset
df_capped = pd.read_csv(capped_path)

# Drop label column for PCA
X = df_capped.drop(columns=['target'])

# Apply PCA
pca = PCA().fit(X)

# Scree plot: cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linewidth=2)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot: PCA Explained Variance')
plt.grid(True)

# Highlight 95% threshold
threshold_index = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.axvline(x=threshold_index, color='g', linestyle='--', label=f'{threshold_index} Components')

plt.legend()
plt.tight_layout()
plt.show()