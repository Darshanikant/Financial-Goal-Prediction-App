# Financial-Goal-Prediction-App
Financial Goal Prediction App
- A machine learning-based web application developed using Streamlit for predicting financial goals based on user inputs such as expenses, income, savings, and other related factors.

# Features:
- Predict financial goals based on user inputs.
- Visualize financial data with interactive charts.
- Data overview with descriptive statistics.
# Demo:
- You can access the live demo of the application here.

# Table of Contents:
- Installation
- Usage
- App Overview
- Data Visualizations
- Model Training

## Usage
- Once the app is running, you will be presented with the following options:

## Overview:

- Provides a preview of the dataset and basic statistics.
## Prediction:

- Input various financial parameters such as monthly income, savings, expenses, and more.
The app predicts the financial goal as a percentage based on the input.
## Visualize Data:

- Select numerical columns to generate interactive line charts to visualize the dataset.
## Steps to Make Predictions:
- Choose the Prediction option from the sidebar.
## Enter the following information:
- Category (e.g., "Food", "Entertainment", etc.)
- Amount Spent
- Monthly Income
- Savings
- Monthly Expense
- Loan Payment
- Credit Card Bill
- Number of Dependents
- Click on Predict to get the predicted financial goal.
## App Overview
- This app uses a Linear Regression model trained on historical financial data to predict the financial goal based on user input. The model has been trained and saved in the financial goal.pkl file. It also uses a Standard Scaler to preprocess the data before making predictions.

## Components:
- Streamlit Frontend: User interface for data input and visualization.
- Linear Regression Model: Trained model for predicting the financial goal.
- Label Encoding: Categorical data is encoded using label encoding before making predictions.
- Data Scaling: Input data is scaled using StandardScaler to ensure accuracy.
## Data Visualizations
- The app allows you to visualize numerical data with interactive line charts. You can select multiple columns to generate a line chart that shows trends across the dataset.

## How to Use:
- Select the Visualize Data option from the sidebar.
- Choose numerical columns for visualization.
- The app will generate a line chart for the selected columns.
## Model Training
- The Linear Regression model is used for predicting financial goals. Here is a brief explanation of how the model is trained:

## Data Preprocessing:

- The dataset is preprocessed by encoding categorical columns using LabelEncoder.
- Numeric columns are scaled using StandardScaler.
## Training:

- The model is trained using the LinearRegression class from scikit-learn.
- The dataset is split into training and test sets using train_test_split.
## Model Saving:

- Once the model is trained, it is saved using pickle (financial goal.pkl).
- The scaler used for input scaling is also saved as scaler.pkl.
- To retrain the model, modify the train.py script, fit the model on your data, and save the new model and scaler files.

