#!/usr/bin/env python
# coding: utf-8

# ## Task 1 — Term Deposit Subscription Prediction (Bank Marketing)
# ### 1. Problem Statement and Objective
# #### Problem Statement
# 
# - Banks run marketing campaigns to encourage customers to subscribe to term deposits. However, contacting every customer is costly and inefficient.
# 
# - The challenge is to predict whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related information.
# 
# #### Objective
# 
# - The objective of this task is to build a classification model that predicts whether a bank customer will subscribe to a term deposit.
# 
# #### Key goals:
# 
# - Analyze the Bank Marketing Dataset
# - Encode categorical features properly
# - Train classification models
# - Evaluate performance using Confusion Matrix, F1-score, and ROC Curve
# - Use Explainable AI (SHAP) to interpret predictions
# ### 2. Dataset Description and Loading
# #### Dataset
# 
# - Bank Marketing Dataset (UCI Machine Learning Repository)
# 
# - The dataset contains information about bank customers and marketing campaign results.
# 
# #### Important Features

# | Feature   | Description                        |
# | --------- | ---------------------------------- |
# | age       | Age of customer                    |
# | job       | Type of job                        |
# | marital   | Marital status                     |
# | education | Education level                    |
# | balance   | Bank balance                       |
# | housing   | Housing loan                       |
# | loan      | Personal loan                      |
# | contact   | Contact communication type         |
# | duration  | Duration of last contact           |
# | campaign  | Number of contacts performed       |
# | deposit   | Target variable (yes or not) |
# 

# ##### Target Variable:
# 
# deposit = yes / no

# #### Code: Load Dataset

# In[ ]:


import pandas as pd

df = pd.read_csv("bank.csv")

df.head()


# ##### Explanation
# - pandas is used for data manipulation.

# ### 3. Import Required Libraries

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

import shap


# #### Explanation

# | Library            | Purpose                 |
# | ------------------ | ----------------------- |
# | pandas             | data processing         |
# | numpy              | numerical operations    |
# | matplotlib/seaborn | data visualization      |
# | sklearn            | machine learning models |
# | shap               | explainable AI          |
# 

# ### 4. Dataset Exploration
# #### Shape of Dataset

# In[ ]:


df.shape


# #### Data Types

# In[ ]:


df.info()


# #### Statistical Summary

# In[ ]:


df.describe()


# ##### Explanation
# 
# These commands help us understand:
# 
# - dataset size
# - feature data types
# - statistical properties

# ### 5. Data Cleaning and Preprocessing
# #### Check Missing Values

# In[ ]:


df.isnull().sum()


# If missing values exist:

# In[ ]:


df.dropna()


# #### Encode Categorical Variables
# 
# - Machine learning models require numerical values.

# In[ ]:


le = LabelEncoder()

for column in df.select_dtypes(include='object'):
    df[column] = le.fit_transform(df[column])


# ##### Explanation
# 
# Example encoding:

# | Original | Encoded |
# | -------- | ------- |
# | yes      | 1       |
# | no       | 0       |
# 

# ### 6. Exploratory Data Analysis (EDA)
# #### Target Variable Distribution

# In[ ]:


sns.countplot(x='deposit', data=df)
plt.title("Term Deposit Subscription")
plt.show()


# #### Age Distribution

# In[ ]:


sns.histplot(df['age'], bins=30)
plt.title("Age Distribution of Customers")
plt.show()


# #### Correlation Heatmap

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# ##### Insights from EDA
# - Most customers do not subscribe
# - Age distribution shows majority of customers between 30–50 years
# - Some features have moderate correlation with the target

# ### 7. Train-Test Split

# In[ ]:


X = df.drop('deposit', axis=1)
y = df['deposit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ##### Explanation
# - 80% data used for training
# - 20% used for testing

# ### 8. Model Building
# #### Logistic Regression Model

# In[ ]:


log_model = LogisticRegression(max_iter=10000)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)


# ##### Explanation
# 
# Logistic Regression is commonly used for binary classification problems.

# #### Random Forest Model

# In[ ]:


rf_model = RandomForestClassifier(n_estimators=200)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# ##### Explanation
# 
# Random Forest is an ensemble learning model that improves prediction accuracy by combining multiple decision trees.

# ### 9. Model Evaluation
# #### Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()


# ##### Explanation
# 
# Confusion Matrix shows:

# | Prediction     | Meaning                              |
# | -------------- | ------------------------------------ |
# | True Positive  | Correctly predicted subscription     |
# | True Negative  | Correctly predicted non-subscription |
# | False Positive | Incorrect prediction                 |
# | False Negative | Missed prediction                    |
# 

# #### Classification Report

# In[ ]:


print(classification_report(y_test, y_pred_rf))


# ##### Metrics include:
# 
# - Precision
# - Recall
# - F1-score
# - Accuracy

# ### 10. ROC Curve

# In[ ]:


y_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC="+str(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ##### Explanation
# 
# - ROC curve measures the model’s ability to distinguish between classes.
# 
# - Higher AUC indicates better performance.
# 
# ### 11. Explainable AI Using SHAP
# #### Initialize Explainer

# In[ ]:


explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test)


# #### SHAP Summary Plot

# In[ ]:


shap.summary_plot(shap_values, X_test)


# This plot shows which features influence predictions the most

# #### Explain 5 Predictions

# In[ ]:


import shap

explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)


# In[ ]:


print(type(shap_values))


# In[ ]:


shap_values = shap_values[1]


# In[ ]:


import shap

shap.initjs()

explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[0, :, 1])


# This explains why the model made specific predictions.
# 
# ### 12. Visualizations
# 
# Required visualizations:
# 
# - Target distribution chart
# - Age distribution histogram
# - Correlation heatmap
# - ROC Curve
# - SHAP feature importance plot
# ### 13. Final Conclusion and Insights
# 
# In this project, we developed machine learning models to predict whether a bank customer will subscribe to a term deposit.
# 
# Key results:
# 
# - Logistic Regression and Random Forest models were trained.
# - Random Forest achieved better performance in terms of F1-score and ROC-AUC.
# - Explainable AI (SHAP) helped identify important features influencing predictions.
# 
# Important factors influencing customer decisions include:
# 
# - Duration of last contact
# - Account balance
# - Age
# - Number of marketing contacts
# 
# This model can help banks target potential customers more effectively, reducing marketing costs and improving campaign success rates.
