# Pima Indian Diabetes Prediction: Exploratory Data Analysis and Regression Modeling - with MLFlow and Docker

## Project Overview
This project explores and predicts diabetes in the Pima Indian population using the Pima Diabetes Dataset from the National Institute of Diabetes and Digestive and Kidney Diseases. The focus is on conducting Exploratory Data Analysis (EDA), identifying key patterns, handling data quality issues, and applying machine learning models to predict the likelihood of diabetes.

## Dataset Information
- **Dataset Source**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Observations**: 750
- **Variables**: 9
- **Target Variable**: Outcome (Diabetes present: 1, Diabetes absent: 0)

## Objectives
1. **Exploratory Data Analysis (EDA)**: 
   - Univariate, Bivariate, and Multivariate analysis to examine distributions, outliers, and correlations.
   - Visualizations including histograms, KDE plots, count plots, box plots, heatmaps, and pair plots.
   
2. **Data Preprocessing**:
   - Handling missing and invalid values in features such as glucose, blood pressure, and BMI.
   - Imputation of missing values using mean/median techniques.

3. **Feature Engineering**:
   - Creation of new features like `SevenOrMorePregnancies` to enhance model prediction capabilities.

4. **Regression Modeling**:
   - Logistic regression model to predict diabetes, based on the number of pregnancies and other features.
   - Evaluation of model performance using accuracy, precision, and F1 score.

## Key Features
- **Data Cleaning**: Addressing issues such as missing data and imbalanced classes.
- **Model Evaluation**: Confusion matrix, accuracy, precision, recall, and F1 score.
- **Model Comparisons**: Logistic Regression, KNN, Decision Trees, Random Forest, and Gradient Boost models were evaluated, with Logistic Regression providing the best results.

## Results
- **Model Performance**:
   - Logistic Regression achieved a testing accuracy of 82% and a precision of 74.51%.
   - F1 score: 0.738, balancing precision and recall effectively.

- **Predictions on Unseen Data**:
   - The logistic regression model was used to predict probabilities of diabetes based on unseen data.

## Installation & Usage
### Requirements:
- Python 3.9
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Install Dependencies
```bash
pip install requirements.txt
