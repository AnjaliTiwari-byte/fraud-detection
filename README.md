# Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using Python and Scikit-learn.

## Overview

Credit card fraud is a major challenge in financial security. This project uses Logistic Regression to classify transactions as legitimate or potentially fraudulent based on historical transaction data.

## Features

* Credit card transaction classification
* Data preprocessing with StandardScaler
* Logistic Regression machine learning model
* Class imbalance handling using balanced class weights
* Model evaluation using precision, recall, F1-score, and ROC-AUC
* Confusion matrix for evaluating classification results
* Saved model for future predictions

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the extracted `creditcard.csv` file in the project folder before training.

The target column is `Class`, where 0 represents a legitimate transaction and 1 represents a fraudulent transaction.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Python script:

```bash
python fraud_detection.py
```

Choose option 1 to train the model and option 2 to test a transaction after training.

## Project Structure

```text
credit-card-fraud-detection/
├── fraud_detection.py
├── requirements.txt
└── README.md
```

## Evaluation

The model prints a classification report, confusion matrix, and ROC-AUC score. Results depend on the dataset and model configuration.

For fraud detection, recall and precision for the fraud class are particularly important because missed fraud and false alarms have different consequences.

## Limitations

This is an educational prototype. Predictions depend on the training data and selected features. It has not been validated for real-world banking decisions.

## Future Improvements

* Add a Streamlit dashboard
* Compare Logistic Regression with Random Forest
* Add exploratory data analysis and visualizations
* Tune the classification threshold
* Track precision-recall performance

## Disclaimer

This project is intended for educational and portfolio purposes only.
