# EnergyGuard AI — Project Report

> Complete this report after running the pipeline on the actual datasets. Do not invent or estimate model results.

## Title
EnergyGuard AI: Electricity Consumption Forecasting, Anomaly Detection and Theft-Risk Screening

## Abstract
Describe the problem, the two datasets, the three ML tasks, the algorithms implemented, the evaluation approach and the main measured results. State that theft-risk predictions are screening signals only.

## 1. Introduction
- Electricity demand varies over time and utilities need ways to understand consumption patterns.
- Unusual readings can be prioritised for review, but an anomaly is not proof of theft.
- Project objectives and scope.

## 2. Dataset Description
### 2.1 UCI Individual Household Electric Power Consumption
Record the downloaded source, date range, dimensions, fields, missingness and data-quality findings from reports/dataset_audit.json.

### 2.2 SGCC Electricity Theft Detection
Record the source, consumer count, number of daily columns, FLAG class distribution, missingness and any excluded consumers. Keep the UCI and SGCC datasets separate.

## 3. Data Preprocessing
Explain timestamp parsing, numeric conversion, treatment of '?' and blanks, short internal interpolation, hourly aggregation, the 54-valid-readings rule, SGCC daily-column ordering, consumer-level feature engineering and training-only imputation.

## 4. Feature Engineering
List and explain UCI lag, rolling and calendar features. List SGCC consumer-level statistics. Explain why consumer identifiers are excluded from model features.

## 5. Train / Validation / Test Design
Describe the chronological UCI split and the consumer-level stratified SGCC split. Include actual sample counts from the generated metrics. Explain how preprocessing is fitted on training data only.

## 6. Models
- Linear Regression — supervised regression baseline.
- Random Forest Regressor — supervised non-linear regression.
- Isolation Forest — unsupervised unusual-pattern detection.
- Random Forest Classifier — supervised binary classification using SGCC FLAG.

## 7. Results and Evaluation
### 7.1 Forecasting
Insert MAE, RMSE and R² for the held-out test period. Include reports/forecast_test.png and discuss errors in kWh.

### 7.2 Anomaly Detection
Report the number and percentage flagged and inspect representative timestamps. Do not report supervised precision/recall because UCI has no anomaly ground-truth labels.

### 7.3 Theft-Risk Classification
Insert precision, recall, F1, PR-AUC, confusion matrix and class-wise results. Discuss the class imbalance and threshold selected on validation data.

## 8. Dashboard
Describe the overview, forecast, anomaly review, theft-risk screening and evaluation pages. Include screenshots from the running Streamlit application.

## 9. Limitations and Responsible Use
- UCI represents one household in France and does not establish population-wide performance.
- SGCC is Chinese utility data; it may not generalise to Indian consumers or different meter systems.
- Missingness and class imbalance can affect model quality.
- Isolation Forest flags unusual patterns, not theft.
- False positives require independent technical/physical investigation.
- Never use model output for automatic disconnection, penalties or enforcement.

## 10. Conclusion and Future Work
Summarise measured outcomes without overstating generalisation. Suggest validation on representative local data, calibrated thresholds, human review and monitoring before any real-world use.

## References
1. UCI Machine Learning Repository, Individual Household Electric Power Consumption: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
2. HenryRDlab, ElectricityTheftDetection: https://github.com/henryRDlab/ElectricityTheftDetection
3. Zheng et al. (2018), Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids, IEEE Transactions on Industrial Informatics. DOI: https://doi.org/10.1109/TII.2017.2785963
