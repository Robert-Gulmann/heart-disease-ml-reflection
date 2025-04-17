{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue233;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c0\c93333;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Heart Disease ML Reflection Project\
\
This repository contains the full implementation, evaluation, and reflection for a machine learning project focused on predicting heart disease using the Cleveland Heart Disease dataset.\
\
## \uc0\u55357 \u56513  Project Structure\
\
heart-disease-ml-reflection/\
\uc0\u9500 \u9472 \u9472  Data/                 # Cleaned datasets (excluded via .gitignore)\
\uc0\u9500 \u9472 \u9472  Models/               # Model scripts: Logistic Regression, KNN, Decision Tree\
\uc0\u9500 \u9472 \u9472  Notebooks/            # Preprocessing, PCA, Boxplot, ScreePlot scripts\
\uc0\u9500 \u9472 \u9472  .gitignore            # Ignores .csv, outputs, and system files\
\uc0\u9492 \u9472 \u9472  README.md             # Project overview (this file)\
\
## \uc0\u55356 \u57263  Objective\
\
To build and evaluate three supervised ML models:\
- Logistic Regression\
- K-Nearest Neighbors (KNN)\
- Decision Tree\
\
Each model was tested:\
- On raw vs capped datasets\
- With and without PCA\
\
## \uc0\u55357 \u56522  Evaluation Metrics\
\
- Accuracy\
- F1 Score\
- ROC-AUC\
- Confusion Matrix\
\
## \uc0\u55357 \u56481  Key Findings\
\
- **KNN with PCA** had the highest AUC (0.93)\
- **Logistic Regression** was the most interpretable and clinically reliable\
- **Decision Trees** offered good explainability but tended to overfit\
\
## \uc0\u55357 \u56514  Data\
\
The data is based on the Cleveland Heart Disease dataset, with three versions:\
- Heart-Disease-Dataset.csv\
- HeartDisease_df_original.csv\
- HeartDisease_df_capped.csv\
\
(Note: CSV files are excluded from the repo via `.gitignore`)\
\
## \uc0\u55358 \u56800  Notebooks & Scripts\
\
Scripts for each modelling step are located in:\
- `/Notebooks` for preprocessing and PCA\
- `/Models` for classification scripts\
\
## \uc0\u55357 \u56541  Full Report\
\
The full reflective report is submitted separately and discusses:\
- Model performance\
- Clinical applicability\
- Ethical considerations\
- Limitations and next steps\
\
## \uc0\u55357 \u56599  GitHub Link\
\
https://github.com/Robert-Gulmann/heart-disease-ml-reflection\
\
---\
\
### \uc0\u55357 \u56550  Cloning Instructions\
\
\pard\pardeftab720\partightenfactor0

\f1 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 git clone {\field{\*\fldinst{HYPERLINK "https://github.com/Robert-Gulmann/heart-disease-ml-reflection.git"}}{\fldrslt \cf3 \ul \ulc3 \strokec3 https://github.com/Robert-Gulmann/heart-disease-ml-reflection.git}}}