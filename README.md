# Cardiovascular Disease Analysis and Prediction

This repository contains a comprehensive analysis and machine learning pipeline to predict the presence of heart disease using a dataset related to cardiovascular health and various other lifestyle factors. The project covers data cleaning, visualization, modeling, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Cleaning](#data-cleaning)
- [Data Visualization](#data-visualization)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Overview

This project analyzes a dataset (`CVD_cleaned.csv`) with information on cardiovascular disease (CVD) risk factors, including:
- Demographic features (e.g., Age, Sex)
- Lifestyle factors (e.g., Exercise, Smoking)
- Health conditions (e.g., Diabetes, Arthritis, Heart Disease)

It builds a machine learning model using XGBoost to predict the presence of heart disease. Additionally, extensive exploratory data analysis (EDA) is performed to understand feature distributions, relationships, and correlations.

## Project Structure

```bash
├── CVD_cleaned.csv         # Dataset for analysis
├── Cardio_Vascular_Disease_Risk_Prediction.py             # Python script with data cleaning, visualization, and modeling code
├── README.md               # Project documentation
└── requirements.txt        # Required Python packages

## Data Cleaning

- Handling categorical variables: Categorical variables were encoded using techniques such as one-hot encoding and label encoding.
- Mapping: Variables like General_Health, BMI_Category, and Age_Category were mapped into numerical/ordinal values.
- Handling duplicates: Duplicate rows were removed to ensure data integrity.

## Data Visualization

Several types of visualizations were performed, including:

- Univariate Analysis: Distribution of individual features using histograms and count plots.
- Bivariate Analysis: Relationship between features like exercise, smoking, and disease conditions.
- Multivariate Analysis: 3D scatter plots and correlation heatmaps to explore complex relationships between multiple variables.
- The visualizations help uncover patterns and relationships between different factors contributing to cardiovascular diseases.

## Modeling and Evaluation

- Model: XGBoost classifier was used due to its effectiveness in handling class imbalances.
- Resampling: SMOTE and Tomek Links were used to address imbalances in the dataset (oversampling the minority class and undersampling the majority class).
- Evaluation: The model was evaluated using metrics like precision, recall, F1-score, ROC-AUC, and confusion matrices. Both ROC and precision-recall curves were plotted to assess model performance.

## Installation

- Prerequisites
- Python 3.7+

## Usage

- Place your dataset (CVD_cleaned.csv) in the root folder.
- Run the Cardio_Vascular_Disease_Risk_Prediction.py script to perform the analysis and model training:

Results
- The model achieves 21% precision and 77% recall for predicting heart disease, with an ROC-AUC score of 0.75.
- The analysis provides insights into how factors like BMI, exercise, and age are related to cardiovascular diseases.
- See the visualizations generated during analysis for detailed patterns and trends.
