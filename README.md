# Term Deposit Subscription Prediction

## Objective
Predict whether a bank customer will subscribe to a term deposit based on marketing campaign data. The goal is to help banks target potential customers more effectively and improve campaign efficiency.

## Dataset
**Bank Marketing Dataset** (UCI Machine Learning Repository)  
Contains demographic, financial, and marketing-related information for customers.

### Key Features
- `age` – Customer age  
- `job` – Job type  
- `marital` – Marital status  
- `education` – Education level  
- `balance` – Account balance  
- `housing` – Housing loan  
- `loan` – Personal loan  
- `contact` – Contact communication type  
- `duration` – Duration of last contact  
- `campaign` – Number of contacts  
- `y` – Target variable (Yes = subscribed, No = not subscribed)

## Approach
1. Data Cleaning & Preprocessing  
   - Handling missing values  
   - Encoding categorical variables  
2. Exploratory Data Analysis (EDA)  
   - Visualizations for understanding patterns and correlations  
3. Model Building  
   - Logistic Regression  
   - Random Forest Classifier  
4. Model Evaluation  
   - Confusion Matrix  
   - F1-score  
   - ROC Curve & AUC  
5. Explainable AI  
   - SHAP analysis to interpret predictions

## Tools & Technologies
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- SHAP (Explainable AI)

## Results
- Random Forest performed better than Logistic Regression.  
- SHAP analysis identified key features influencing subscription, including `duration`, `balance`, and `age`.  
- Banks can use this model to target likely subscribers and reduce marketing costs.

## How to Run
1. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
