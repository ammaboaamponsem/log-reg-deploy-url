# Income Prediction Microservice
## Overview

A machine learning microservice that predicts individual income brackets (>50K or ≤50K) using Census demographic data. The model analyzes 14 features including education, occupation, and demographic factors to estimate income levels with 80.28% accuracy.

## Live Demo
Access the live prediction service: Income Predictor App (https://log-reg-deploy-url-79k4lv2pp4vdcdzffg6a4v.streamlit.app/)

## Features
Real-time income predictions

Probability scoring

User-friendly web interface

No authentication required

14 input parameters support

## Technical Details
Model: Logistic Regression

Accuracy: 80.28%

Precision: 67.61%

Recall: 39.88%

F1 Score: 50.17%

## Tech Stack

Python

scikit-learn

Pandas

Streamlit

GitHub Actions

## Installation
bashCopygit clone [repository-url]

pip install -r requirements.txt

streamlit run app.py

## Dataset
Adult/Census Income dataset from UCI Machine Learning Repository, featuring 32,561 instances with 14 attributes.
