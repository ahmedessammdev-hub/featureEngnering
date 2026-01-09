# 🚢 Titanic: Missing Values & Feature Engineering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project demonstrating **missing value imputation** and **feature engineering** techniques applied to the famous Titanic dataset. This project showcases real-world data preprocessing pipelines using scikit-learn.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Techniques](#key-techniques)
- [Project Workflow](#project-workflow)
- [Code Examples](#code-examples)
- [Results](#results)
- [Installation](#installation)
- [Author](#author)

## 📊 Overview

This project demonstrates a complete end-to-end machine learning pipeline:

✅ **Data Loading** - Loading the Titanic dataset using Seaborn  
✅ **Exploratory Data Analysis** - Understanding missing values and distributions  
✅ **Missing Value Handling** - Implementing imputation strategies  
✅ **Feature Engineering** - Processing and transforming features  
✅ **Model Building** - Training a Logistic Regression classifier  
✅ **Evaluation** - Assessing model performance with metrics

## 🚢 Dataset

The **Titanic Dataset** is a famous dataset containing information about 891 passengers from the RMS Titanic. The goal is to predict whether a passenger survived the sinking.

### Features Overview

| Feature | Type | Description |
|---------|------|-------------|
| **pclass** | Categorical | Passenger class (1, 2, or 3) |
| **sex** | Categorical | Gender of passenger |
| **age** | Numerical | Age in years (contains missing values) |
| **sibsp** | Numerical | Number of siblings/spouses aboard |
| **parch** | Numerical | Number of parents/children aboard |
| **fare** | Numerical | Ticket price (contains missing values) |
| **embarked** | Categorical | Port of embarkation (contains missing values) |
| **survived** | Binary Target | 0 = Did not survive, 1 = Survived |

### Data Visualizations

<table>
<tr>
<td><b>Age Distribution</b></td>
<td><b>Embarked Distribution</b></td>
</tr>
<tr>
<td><img src="age_plot.png" alt="Age Plot" width="400"/></td>
<td><img src="embarked_plot.png" alt="Embarked Plot" width="400"/></td>
</tr>
</table>

## 🔧 Key Techniques

### 1. Missing Value Detection 📍

First, we identify which columns have missing values:

```python
# Check for missing values
X.isnull().sum()

# Output shows:
# age       177 missing values
# fare      0 missing values
# embarked  2 missing values
```

### 2. Handling Missing Values with Imputation 🔄

**For Numerical Features (Age, Fare):**
```python
number_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill with median
    ('StandardScaler', StandardScaler())             # Normalize values
])
```

**For Categorical Features (Embarked, Sex, Pclass):**
```python
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill with mode
    ('encoder', OneHotEncoder(handle_unknown='ignore'))     # Convert to binary
])
```

### 3. Unified Preprocessing Pipeline 🔗

Combine all preprocessing steps using `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ('num', number_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])
```

### 4. Complete ML Pipeline 🚀

Build an end-to-end pipeline combining preprocessing and model:

```python
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=100000))
])

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
```

## 📊 Project Workflow

```
┌─────────────────────┐
│   Load Dataset      │ ← Titanic from Seaborn
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Explore Data       │ ← Check missing values & distributions
└──────────┬──────────┘
           │
┌──────────▼──────────────────┐
│  Split Data (75/25)         │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────────────────┐
│  Create Preprocessing Pipeline          │
│  ├─ Numerical: Impute → Scale          │
│  └─ Categorical: Impute → Encode       │
└──────────┬──────────────────────────────┘
           │
┌──────────▼──────────────────┐
│  Train Model                │ ← Logistic Regression
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│  Evaluate Performance       │ ← Accuracy & Metrics
└─────────────────────────────┘
```

## 💻 Code Examples

### Loading and Exploring Data
```python
import pandas as pd
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display first rows
df.head()

# Select features and target
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

# Split with 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=0.75, 
    random_state=22
)
```

### Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report

# Get predictions
y_pred = model.predict(x_test)

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Technologies Used

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning and preprocessing
- **seaborn**: Data visualization and dataset loading
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computations

## � Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Install Dependencies

```bash
pip install pandas scikit-learn seaborn matplotlib numpy
```

### 2. Run the Notebook

```bash
# Start Jupyter
jupyter notebook missing_values.ipynb
```

### 3. Execute All Cells

Press `Ctrl+Shift+Enter` or use the Run All option to execute the complete pipeline.

## 📚 Technology Stack

| Technology | Purpose |
|-----------|---------|
| 🐼 **Pandas** | Data manipulation and analysis |
| 📊 **Seaborn** | Data visualization and dataset loading |
| 🤖 **Scikit-learn** | Machine learning and preprocessing |
| 📈 **Matplotlib** | Advanced plotting and visualization |
| 🔢 **NumPy** | Numerical computations |

## 🎓 Learning Outcomes

By studying this project, you'll learn:

✅ How to handle **missing data** in real-world datasets  
✅ How to build **robust preprocessing pipelines**  
✅ Best practices to avoid **data leakage**  
✅ How to evaluate **classification models**  
✅ How to combine multiple preprocessing steps using **pipelines**  
✅ Techniques for **feature engineering** and transformation

## 📂 Project Structure

```
featureEngnering/
├── missing_values.ipynb       # Main notebook with complete analysis
├── README.md                  # This file
├── age_plot.png               # Age distribution visualization
└── embarked_plot.png          # Embarked distribution visualization
```

## 👤 Author

**Ahmad Essam**  
*Machine Learning & Data Science Enthusiast*

---

<div align="center">

### 🌟 If you found this project helpful, please give it a star!

</div>
