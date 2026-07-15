# AI Data Scientist

> An end-to-end AI-powered Data Science platform that automates the complete machine learning workflow — from raw dataset upload to business-ready insights.

---

# Project Overview

Building a machine learning model usually requires many repetitive steps.

Every time a new dataset arrives, a data scientist has to:

- Understand the dataset
- Check missing values
- Perform Exploratory Data Analysis (EDA)
- Create useful features
- Preprocess the data
- Train multiple machine learning models
- Compare model performance
- Explain predictions using SHAP
- Select important features
- Retrain the optimized model
- Generate business insights
- Prepare reports for stakeholders

These steps are repeated for almost every project, regardless of the dataset.

This process takes a significant amount of time and requires strong knowledge of data science.

The goal of **AI Data Scientist** is to automate this entire workflow with a single application.

Users only need to upload their dataset, and the platform automatically performs the complete machine learning pipeline while generating business-friendly insights.

---

# Problem Statement

Organizations generate huge amounts of data every day.

However, converting raw data into useful business decisions is still a slow and manual process.

A data scientist spends hours on repetitive tasks such as:

- Cleaning data
- Finding missing values
- Performing exploratory analysis
- Creating features
- Selecting algorithms
- Comparing models
- Explaining predictions
- Writing reports

For small companies, startups, students, and non-technical users, hiring a complete data science team is often expensive.

There is a need for a system that can automate these repetitive tasks while still producing understandable and reliable results.

---

# Solution

AI Data Scientist automates the complete machine learning workflow.

Instead of manually writing hundreds of lines of code, users simply upload a dataset.

The application then automatically:

- Understands the dataset
- Detects data quality issues
- Performs Exploratory Data Analysis
- Creates useful features
- Preprocesses the data
- Detects the machine learning problem automatically
- Trains multiple ML models
- Selects the best-performing model
- Explains predictions using SHAP
- Selects the most important features
- Retrains an optimized model
- Generates AI-powered business insights
- Creates a downloadable PDF report

---

# Why I Built This Project

While working on different machine learning projects, I noticed that almost every project followed the same workflow.

Although the datasets changed, the overall pipeline remained nearly identical.

I wanted to build a system that could automate these repetitive tasks so that users could focus more on solving business problems instead of writing boilerplate code.

The objective was not just to build another AutoML tool but to create an AI assistant that behaves like a junior data scientist.

---

# Features

## Dataset Upload

- CSV Support
- Excel Support

---

## Dataset Overview

Automatically displays:

- Number of rows
- Number of columns
- Missing values
- Duplicate records
- Dataset preview
- Column information
- Statistical summary

---

## Dataset Health Report

Automatically checks:

- Missing values
- Duplicate rows
- Constant columns
- High-cardinality features
- Highly correlated features

---

## Exploratory Data Analysis

Automatically generates:

- Missing value analysis
- Numerical feature distributions
- Correlation heatmap
- Categorical analysis
- Basic AI-generated insights

---

## Automatic Feature Engineering

The platform automatically performs feature engineering before model training.

Examples include:

- Data cleaning
- Feature transformations
- Correlation handling
- Better feature preparation

---

## Automatic Problem Detection

The system automatically identifies whether the dataset is:

- Classification
- Regression

No manual selection is required.

---

## Multiple Model Training

For Classification:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- KNN
- Support Vector Machine
- Naive Bayes
- AdaBoost
- Gradient Boosting
- Extra Trees

For Regression:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Ridge Regression
- Lasso Regression
- ElasticNet
- AdaBoost Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

The application automatically compares every model and selects the best one.

---

## Explainable AI (SHAP)

Instead of treating the model as a black box, Evo explains:

- Which features matter most
- Why predictions are made
- Feature importance visualization

---

## SHAP-Based Feature Selection

The application automatically:

- Identifies important features
- Removes less useful features
- Creates an optimized feature set

---

## Optimized Model Retraining

Using only important features, the platform:

- Retrains the model
- Compares performance
- Shows whether optimization improved the results

---

## AI Business Analyst

Using Google's Gemini API, generates:

- Executive Summary
- Dataset Health
- Business Insights
- Model Performance Analysis
- Important Drivers
- Business Recommendations
- Final Conclusion

The report is written in simple business language that non-technical users can understand.

---

## PDF Report Generation

Generate a complete professional report containing:

- Dataset summary
- Model performance
- SHAP explanation
- AI business insights

# Tech Stack

### Frontend

- Streamlit

### Backend

- Python

### Machine Learning

- Scikit-Learn
- XGBoost

### Explainability

- SHAP

### Visualization

- Plotly
- Matplotlib

### Data Processing

- Pandas
- NumPy

### AI

- Google Gemini API
- LangChain

### Report Generation

- ReportLab

---

# Folder Structure

```
AI_Data_Scientist/

│

├── app.py

├── requirements.txt

├── README.md

│

├── src/

│ ├── data_processing.py

│ ├── eda.py

│ ├── preprocessing.py

│ ├── problem_detection.py

│ ├── feature_engineering.py

│ ├── model_training.py

│ ├── explainability.py

│ ├── feature_selection.py

│ ├── model_optimization.py

│ ├── model_evaluation.py

│ ├── llm.py

│ └── report_generator.py

│

├── reports/

├── datasets/

└── images/
```

# Usage

1. Upload a CSV or Excel dataset.
2. Review the dataset overview and health report.
3. Select the target column.
4. Train machine learning models.
5. Generate SHAP explanations.
6. Optimize the model using selected features.
7. Generate AI-powered business insights.
8. Download the final PDF report.

---

# Machine Learning Pipeline

```
Upload Dataset

↓

Data Validation

↓

EDA

↓

Feature Engineering

↓

Preprocessing

↓

Problem Detection

↓

Train Multiple Models

↓

Model Comparison

↓

Best Model Selection

↓

SHAP Explainability

↓

Feature Selection

↓

Optimized Retraining

↓

Business Insights

↓

PDF Report
```

---

# Results

The platform can automatically:

- Reduce manual data science effort.
- Train and compare multiple machine learning models.
- Explain predictions using SHAP.
- Identify the most important features.
- Retrain an optimized model.
- Generate executive-level business reports.
- Produce downloadable PDF reports.

---

# Future Improvements

Some planned improvements include:

- Time Series Forecasting
- NLP Dataset Support
- Image Classification Support
- Hyperparameter Optimization
- Model Deployment APIs
- User Authentication
- Database Integration
- Experiment Tracking
- Cloud Storage Support
- Multi-language Business Reports

---

# Contributing

Contributions are always welcome.

If you have ideas for improving AI Data Scientist, feel free to fork the repository, create a new branch, and submit a pull request.

# Author

## Aakarsh Kumar ##
Data Science Intern - Piramal Finance
B.Tech, NIT Allahabad

Passionate about AI, Machine Learning, MLOps, and Generative AI.
---

# Acknowledgements

This project was built using several amazing open-source technologies.

Special thanks to:

- Streamlit
- Scikit-Learn
- SHAP
- XGBoost
- LangChain
- Google Gemini
- Pandas
- NumPy
- Plotly
- Matplotlib
- ReportLab

---
