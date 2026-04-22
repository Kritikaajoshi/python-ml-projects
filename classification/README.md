# ML Classification Pipeline

Binary classification task for counterfeit banknote detection using 8 classifiers.

## Models Compared
Logistic Regression, KNN, SVM, MLP, Decision Tree, Random Forest, XGBoost, Naive Bayes

## Results (on E.csv)

| Model | Accuracy | F1 | MCC |
|---|---|---|---|
| Logistic Regression | 0.9632 | 0.9550 | 0.9267 |
| KNN (k=3) | 1.0000 | 1.0000 | 1.0000 |
| SVM | 1.0000 | 1.0000 | 1.0000 |
| MLP | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 0.9706 | 0.9630 | 0.9390 |
| Random Forest | 0.9412 | 0.9200 | 0.8777 |
| XGBoost | 0.9338 | 0.9091 | 0.8628 |
| Naive Bayes | 0.8824 | 0.8400 | 0.7509 |

## Key Features
- Restored pre-fitted StandardScaler from JSON
- Majority voting consensus across all 8 models
- Evaluated on 6 metrics: Accuracy, Precision, Recall, F1, FDR, MCC
- Logging system outputs to both console and logfile.txt

## Tech
Python, scikit-learn, XGBoost, pandas, NumPy