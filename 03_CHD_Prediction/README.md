# Cardiovascular Disease Prediction

#### Project Overview

This analysis undertakes a comprehensive, data-driven approach to predict the 10-year risk of Coronary Heart Disease (CHD) using the Framingham Heart Study dataset.  

#### Objective
To build and evaluate machine learning model that can effectively identify individuals at high risk for developing future heart disease.  

The workflow is structured across three phases:   
- [I Data Cleaning](https://github.com/monikase/Data_Science/blob/main/03_CHD_Prediction/CHD_I_DataCleaning.ipynb)
- [II EDA](https://github.com/TuringCollegeSubmissions/msedui-PYDA.4.4/blob/main/CHD_II_EDA.ipynb)
- [III Modeling](https://github.com/TuringCollegeSubmissions/msedui-PYDA.4.4/blob/main/CHD_III_Modeling.ipynb)

#### Model Training and Tuning:   
Two models were selected: Logistic Regression and Random Forest. Both were optimized using GridSearchCV to find the best-performing hyperparameters.

#### Key Takeaways

- Recall, which measures the model's ability to find all actual positive cases, was the most critical evaluation metric.
- **The Logistic Regression model** with balanced class weights achieved the highest **recall of 69.6%**.
- Key Risk Factors Identified:
  - Age – The strongest predictor; older patients have substantially higher risk.
  - Gender – Males have higher early CHD risk than females.
  - Smoking (cigsPerDay) – Current smokers and heavier smoking strongly increase risk.
  - Blood Pressure – Higher pulse pressure indicate elevated risk.
