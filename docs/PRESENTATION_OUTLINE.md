# EnergyGuard AI — Presentation Outline

Use actual outputs from the completed training run. Do not insert fabricated accuracy or performance values.

1. **Title** — EnergyGuard AI; team / course details.
2. **Problem Statement** — consumption planning, unusual patterns and review prioritisation.
3. **Objectives and Scope** — forecasting, anomaly detection, theft-risk screening and Streamlit dashboard; no live utility integration or enforcement.
4. **Datasets** — UCI household data and SGCC consumer data; explain why they remain separate.
5. **Preprocessing** — missing values, valid zeros, hourly aggregation, consumer-level features and leakage controls.
6. **Feature Engineering** — UCI lag/rolling/calendar features; SGCC daily consumption statistics.
7. **Algorithms** — Linear Regression, Random Forest Regressor, Isolation Forest and Random Forest Classifier.
8. **System Workflow** — raw data → audit → preprocessing → feature engineering → splits → training/evaluation → dashboard.
9. **Forecasting Results** — MAE, RMSE, R² and actual-vs-predicted plot.
10. **Anomaly Results** — flagged count, flagged percentage and examples; clarify that anomalies are not theft labels.
11. **Theft-Risk Results** — precision, recall, F1, PR-AUC and confusion matrix; explain class imbalance.
12. **Dashboard Demo** — overview, forecast, anomaly review, risk screening and evaluation.
13. **Limitations and Responsible Use** — geographic generalisation, missingness, false positives and need for human investigation.
14. **Conclusion and Future Scope** — evidence-based summary and possible validation on representative local data.
15. **References** — UCI, SGCC repository and cited research paper.
