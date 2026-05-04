# Interpretable Football Prediction and Analytics System

This repository contains the code and files for an interpretable machine learning framework for predicting English Premier League match outcomes using publicly available football data.

## Project Overview

The system uses historical Premier League match data to create football-specific features such as approximate expected goals, recent form, rolling xG, and shot accuracy. Machine learning models are trained to predict match outcomes as home win, draw, or away win.

The project also includes SHAP explainability to interpret model predictions and a Streamlit dashboard for interactive prediction and visual analytics.

## Main Features

- Data preprocessing and cleaning
- Approximate xG feature engineering
- Rolling last-five-match form features
- Logistic Regression and Random Forest models
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- SHAP feature importance and summary plots
- Streamlit dashboard for prediction and team analysis

## Repository Contents

- FINAL_PROJECT_ANKIT.ipynb - Google Colab notebook containing data cleaning, feature engineering, model training, evaluation, and SHAP analysis.
- app.py - Streamlit dashboard application.
- best_rf_model.pkl - Saved trained Random Forest model.
- epl_week4_model_ready.csv - Final feature-engineered dataset.
- shap_home_win_bar.png - SHAP feature importance plot.
- shap_home_win_summary.png - SHAP summary plot.
- requirements.txt - Python packages required to run the application.

## Dataset

The dataset used in this project is based on publicly available English Premier League match data from Football-Data.co.uk.

The data is used for academic purposes only.

## How to Run the Streamlit App

1. Download or clone this repository.

2. Make sure all files remain in the same folder as app.py, especially:
   - best_rf_model.pkl
   - epl_week4_model_ready.csv
   - shap_home_win_bar.png
   - shap_home_win_summary.png

3. Install the required Python packages by running this command in the terminal:

   pip install -r requirements.txt

4. Run the Streamlit application by running this command in the terminal:

   streamlit run app.py

5. The application should open in your browser at:

   http://localhost:8501

## Important Note

All required files must remain in the same folder as app.py. If any files are moved to another folder, the file paths inside app.py may need to be updated.

## Tools Used

- Python
- Google Colab
- PyCharm
- pandas
- NumPy
- scikit-learn
- SHAP
- Streamlit
- matplotlib
- seaborn
- joblib

## Project Workflow

Football-Data Dataset
        ↓
Data Cleaning and Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning Model Training
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Streamlit Dashboard

## Author

Ankit
