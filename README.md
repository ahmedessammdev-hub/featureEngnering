# Missing Values & Feature Engineering Project

This project demonstrates handling missing values and feature engineering techniques using the Titanic dataset with machine learning.

## Overview

The project showcases a complete pipeline for:
- **Data Loading**: Loading the Titanic dataset using Seaborn
- **Missing Value Handling**: Implementing strategies to handle missing data
- **Feature Engineering**: Processing numerical and categorical features
- **Model Training**: Building and training a Logistic Regression classifier
- **Model Evaluation**: Assessing model performance

## Dataset

The **Titanic Dataset** contains passenger information with the following features:
- **Numerical Features**: age, sibsp, parch, fare
- **Categorical Features**: pclass, sex, embarked
- **Target Variable**: survived (binary classification)

## Key Techniques

### 1. Missing Value Imputation

**Numerical Features:**
- Strategy: Median imputation
- Imputes missing values in 'age' and 'fare' with the median value

**Categorical Features:**
- Strategy: Most frequent imputation
- Imputes missing values with the most common category

### 2. Feature Preprocessing

**Numerical Features:**
- StandardScaler normalization for improved model convergence

**Categorical Features:**
- OneHotEncoder to convert categorical variables to binary format

### 3. Preprocessing Pipeline

Uses `ColumnTransformer` to apply different transformations to numerical and categorical features simultaneously, ensuring consistency between training and test sets.

## Project Structure

```
missing_values.ipynb          # Main notebook with complete analysis
```

## Technologies Used

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning and preprocessing
- **seaborn**: Data visualization and dataset loading
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computations

## Model Performance

The Logistic Regression model trained on the preprocessed data achieves:
- **Accuracy**: Displayed in the notebook output
- **Classification Report**: Includes precision, recall, and F1-score

## Running the Project

1. Ensure all required libraries are installed:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib numpy
   ```

2. Open the notebook:
   ```bash
   jupyter notebook missing_values.ipynb
   ```

3. Run all cells to execute the complete pipeline

## Learning Outcomes

This project demonstrates:
- Proper handling of missing data in real-world datasets
- Building robust preprocessing pipelines
- Implementing best practices to avoid data leakage
- Evaluating classification models
- Combining multiple preprocessing steps in a single pipeline

## Author

Feature Engineering Course Project
