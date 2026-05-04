# Interpretble-Football-Prediction-and-analytics-system
Interpretable machine learning framework for Premier League match outcome prediction using SHAP and Streamlit.


This project is an interpretable machine learning framework for predicting English Premier League match outcomes using publicly available football data.

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

- `FINAL_PROJECT_ANKIT.ipynb` - Google Colab notebook containing data cleaning, feature engineering, model training, evaluation, and SHAP analysis.
- `app.py` - Streamlit dashboard application.
- `best_rf_model.pkl` - Saved trained Random Forest model.
- `epl_week4_model_ready.csv` - Final feature-engineered dataset.
- `shap_home_win_bar.png` - SHAP feature importance plot.
- `shap_home_win_summary.png` - SHAP summary plot.
- `requirements.txt` - Python packages required to run the application.

## Dataset

The dataset used in this project is based on publicly available English Premier League match data from Football-Data.co.uk.

## How to Run the Streamlit App

Install the required packages:

```bash
pip install -r requirements.txt
