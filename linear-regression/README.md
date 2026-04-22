# Linear Regression from Scratch

Implemented 4 varieties of linear regression using only NumPy.
No sklearn modeling functions used.

## What's Implemented
- Closed-form solution (normal equation)
- Full-batch gradient descent
- Stochastic gradient descent
- Mini-batch gradient descent

## Dataset
Baby weights dataset with 101,400 samples and 36 features.
Mixed data types: numeric, ordinal, nominal, binary.

## Preprocessing Pipeline
- Train/test split (75/25)
- Median imputation for numeric features
- OrdinalEncoder for ordinal features
- OneHotEncoder for nominal features
- StandardScaler for numeric features

## Results
| Model | RMSE |
|---|---|
| Closed-form | 55.07 |
| Full-batch GD | 1.11 |
| Stochastic GD | 4.19 |
| Mini-batch GD | 1.42 |

Best model: Full-batch GD (RMSE 1.11)
Best hyperparameters: alpha=0.001, batch_size=32

## Tech
Python, NumPy, pandas, scikit-learn (preprocessing only)