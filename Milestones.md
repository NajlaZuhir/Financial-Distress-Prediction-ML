# Financial-Distress-Predictor

### Milestone 1 – Machine Learning Modeling & Evaluation (CRISP-DM)


Conclusion:

In this milestone, I successfully completed the end-to-end modeling phase of the financial distress prediction problem following the CRISP-DM framework. I prepared and encoded the data, performed feature scaling, and applied plateau-based feature selection using logistic regression coefficients to identify the most informative financial indicators. A final model was trained using frozen optimal features and evaluated on validation and test sets using multiple metrics, including F1-macro, ROC-AUC, and Precision–Recall curves, with a focus on class imbalance. The model achieved stable generalization performance, demonstrating strong ranking ability despite low minority-class precision. This milestone strengthened my understanding of model interpretability, feature sensitivity analysis, and evaluation beyond accuracy, and resulted in a finalized, reproducible ML model ready for deployment.

### Milestone 2 – MLOps & Model Deployment

Conclusion:

In this milestone, I operationalized the trained machine learning model by applying MLOps principles learned in the course. The model, scaler, and selected features were serialized and integrated into a FastAPI-based inference service supporting both single and batch predictions. The application was fully containerized using Docker, enabling consistent and reproducible execution across environments. End-to-end testing was performed via API calls from the terminal to validate model inference, preprocessing, and response formatting. This milestone provided hands-on experience with model serving, API design, artifact management, and containerization, bridging the gap between experimental modeling and production-ready machine learning systems.

### Deliverables for Milestone 3

✔ Logging added to FastAPI
✔ Drift monitoring notebook / script
✔ Model metadata saved with artifacts
✔ Clear retraining criteria
✔ Updated README explaining lifecycle